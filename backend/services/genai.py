import re
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from vertexai.generative_models import GenerativeModel
import json
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
    
    # Cost Analysis
    def count_total_tokens(self, docs: list):
        total = 0
        temp_model = GenerativeModel("gemini-1.0-pro")
        logger.info("Counting total billable characters...")
        
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
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
        total_billable_characters = self.GeminiProcessor.count_total_tokens(result)

        if verbose:
            logger.info(f"Author: {author}")
            logger.info(f"Length: {length}")
            logger.info(f"Title: {title}")
            logger.info(f"Total size: {totol_size}")
            logger.info(f"Total billable characters: {total_billable_characters}")

        return result
    
    def handle_string_to_json(self, input_string):
        try:
            # Remove leading and trailing code block markers
            cleaned_string = input_string.strip('```json').strip('```').strip()
            
            # Attempt to convert the cleaned string to a JSON object
            json_object = json.loads(cleaned_string)
            
            return json_object
        except json.JSONDecodeError:
            logging.warn("One group of documents is not able to convert to JSON format.")
            return False

    def find_key_concepts(self, documents: list, sample_size: int=0, verbose=False):
        # Iterate through all documents of group size N and find key concepts
        if sample_size > len(documents):
            raise ValueError("Group size is larger than the documents amount")
        
        # Optimize sample size given no input
        if sample_size == 0:
            sample_size = len(documents) // 5
            if verbose: logging.info(f"No sample size specified. Setting number of documents per sample as 5. Sample Size: {sample_size}")

        # Find number of documents in each group
        num_docs_per_group = (len(documents) // sample_size) + (len(documents) % sample_size > 0)

        # Check thersholds for response quality
        if num_docs_per_group > 10:
            raise ValueError("Each group has more than 10 documents and output quality will be degraded significantly. Increase the sample_size parameter to reduce the number of documents per group.")
        elif num_docs_per_group > 5:
            logging.warn("Each group has more than 5 documents and output quality is likely to be degraded. Consider increasing the sample size.")

        # Form groups
        groups = [documents[i:i + num_docs_per_group] for i in range(0, len(documents), num_docs_per_group)]

        batch_concepts = []
        batch_cost = 0
        
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
                
                Respond in the following format as a JSON syntax without any backticks separating each concept with a comma:
                {{"concept": "definition", "concept": "definition", "concept": "definition"}}

                Please return the response as plain JSON, without any code block markers, nothing like ```json or ``` at the end.
                """,
                input_variables = ["text"]
            )

            # Create chain
            chain = prompt | self.GeminiProcessor.model

            # Run chain
            output_concept = chain.invoke({"text": group_content})
            
            # Validate JSON and append the json object to batch_concepts result
            cleaned_concept = self.handle_string_to_json(output_concept)
            if cleaned_concept:
                batch_concepts.append(cleaned_concept)

            # Post Processing Observation
            if verbose:
                total_input_char = len(group_content)
                total_input_cost = (total_input_char / 1000) * 0.000125

                logging.info(f"Running chain on {len(group)} documents")
                logging.info(f"Total input characters: {total_input_char}")
                logging.info(f"Total input cost: {total_input_cost}")

                total_output_char = len(output_concept)
                total_output_cost = (total_output_char / 1000) * 0.000375

                logging.info(f"Total output characters: {total_output_char}")
                logging.info(f"Total ouput cost: {total_output_cost}")

                batch_cost += (total_input_cost + total_output_cost)
                logging.info(f"Total group cost: {total_input_cost + total_output_cost}")
        
        logging.info(f"Total Analysis Cost: {batch_cost}")

        
        return batch_concepts
    