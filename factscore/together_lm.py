from factscore.lm import LM
import openai
import time
import os
import pickle
import logging


class TogetherLM(LM):
    def __init__(self, model_name: str, api_key: str, cache_file=None):
        super().__init__(cache_file)
        self.model = model_name
        self.api_key = api_key
        self.client = openai.Client(
            api_key=api_key,
            base_url="https://api.together.xyz/v1"
        )

    def load_model(self):
        pass  # we have nothing to do here

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_output_length
            )
            generated_text = response.choices[0].message['content'].strip()
            logging.info(f"Generated response: {generated_text}")
            return generated_text
        except Exception as e:
            logging.error(f"Error during API call: {str(e)}")
            return ""  # Return an empty string in case of failure
