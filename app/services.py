from typing import List, Dict, Any
import pinecone
from openai import OpenAI
import httpx

class TenderGenerator:
    def __init__(self, openai_client: OpenAI, pinecone_index: pinecone.Index):
        self.client = openai_client
        self.index = pinecone_index
        self.sections = [
            "NOTICE INVITING TENDER",
            "BRIEF INTRODUCTION",
            "INSTRUCTION TO BIDDERS",
            "SCOPE OF WORK",
            "TERMS AND CONDITIONS",
            "PRICE BID"
        ]

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

    def search_similar_sections(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar tender sections"""
        try:
            query_embedding = self.get_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results['matches']
        except Exception as e:
            raise Exception(f"Error searching similar sections: {str(e)}")

    def generate_tender_section(self, section_name: str, project_details: Dict[str, str],
                              similar_sections: List[Dict[str, Any]]) -> str:
        """Generate a specific section of the tender using GPT-4"""
        try:
            context = "\n\n".join([
                f"Example {i+1}:\n{match['metadata']['content']}"
                for i, match in enumerate(similar_sections)
            ])

            prompt = f"""You are an expert tender document writer. Generate the {section_name} section
            for a new tender based on the following project details and example sections.

            Project Details:
            Title: {project_details['title']}
            Location: {project_details['location']}
            Duration: {project_details['duration']} months
            Budget: {project_details['budget']}
            Description: {project_details['description']}

            Similar Examples from Other Tenders:
            {context}

            Please generate a professional and detailed {section_name} section that follows
            the style and format of the examples while being specific to this project.
            The content should be practical, clear, and legally sound."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert tender document generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating tender section: {str(e)}")

    def generate_complete_tender(self, project_details: Dict[str, str]) -> Dict[str, str]:
        """Generate complete tender document"""
        tender_sections = {}
        try:
            for section in self.sections:
                search_query = f"{section} {project_details['title']} {project_details['description']}"
                similar_sections = self.search_similar_sections(search_query)
                section_content = self.generate_tender_section(
                    section,
                    project_details,
                    similar_sections
                )
                tender_sections[section] = section_content

            return tender_sections
        except Exception as e:
            raise Exception(f"Error generating complete tender: {str(e)}")
