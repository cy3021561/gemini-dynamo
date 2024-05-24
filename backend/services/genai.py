from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from vertexai.generative_models import GenerativeModel
from tqdm import tqdm
import logging

# Configure log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiProcessor:
    def __init__(self, model_name, project):
        self.model = VertexAI(model_name=model_name, project=project)

    def generate_documnet_summary(self, documents: list, **args):

        # "stuff" type would use all raw documents, if too many could go beyond the model's context window limit
        # "map_reduce" type -> reduce numbers of documents based on summaries, ex. 20 docs -> 5 summaries -> take these 5 sums to produce final output
        chain_type = "map_reduce" if len(documents) > 10 else "stuff"
        chain = load_summarize_chain(
            llm = self.model,
            chain_type = chain_type,
            **args
        )
        return chain.run(documents)

    def count_total_token(self, docs: list):
        total = 0
        temp_model = GenerativeModel("gemini-1.0-pro")
        logger.info("Counting total tokens...")
        
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_tokens
        return total

    def get_model(self):
        return self.model


class YoutubeProcessor:
    # Retrieve the full transcript
    def __init__(self, gemini_processor: GeminiProcessor):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.GeminiProcessor = gemini_processor
    
    def retrieve_youtube_documents(self, video_url: str, verbose=False):
        # Load the YouTube video transcript using the YouTubeLoader
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)
        
        author = result[0].metadata["author"]
        length = result[0].metadata["length"]
        title = result[0].metadata["title"]
        totol_size = len(result)

        if verbose:
            logger.info(f"Author: {author}")
            logger.info(f"Length: {length}")
            logger.info(f"Title: {title}")
            logger.info(f"Total size: {totol_size}")

        return result
    
    def find_key_concepts(self, documents: list, group_amount: int=2):
        # Iterate through all documents of group size N and find key concepts
        if group_amount > len(documents):
            raise ValueError("Group size is larger than the documents amount")
        
        # Find number of documents in each group
        num_docs_per_group = (len(documents) // group_amount) + (len(documents) % group_amount > 0)

        # Form groups
        groups = [documents[i:i + num_docs_per_group] for i in range(0, len(documents), num_docs_per_group)]

        batch_concepts = []
        
        logger.info("Finding key concepts...")
        for group in tqdm(groups):
            # Combine content of documents per group
            group_content = ""
            for doc in group:
                group_content += doc.page_content

            # Prompt for finding concepts
            prompt = PromptTemplate(
                template = """
                Find and define key concepts or terms found in the text:
                {text}
                
                Respond in the following format as a JSON object without any backticks separating each concept with a comma:
                "concept": "definition"
                """,
                input_variables = ["text"]
            )

            # Create chain
            chain = prompt | self.GeminiProcessor.model

            # Run chain
            concept = chain.invoke({"text": group_content})
            batch_concepts.append(concept)
        
        return batch_concepts
    