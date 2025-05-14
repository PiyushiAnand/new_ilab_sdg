# Standard
import logging

# Third Party
from datasets import Dataset
from tqdm import tqdm
import openai

# Local
from .block import Block
from ..logger_config import setup_logger
from ..registry import BlockRegistry

logger = logging.getLogger(__name__)

DEFAULT_MAX_NUM_TOKENS = 8192


# This is part of the public API.


@BlockRegistry.register("TranslationBlock")
# pylint: disable=dangerous-default-value
class TranslationBlock(Block):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        block_name,
        config_path,
        client,
        output_cols,
        trans_model_id=None,
        source_lang="eng_Latn",
        target_lang="hin_Deva",
        prompt_struct=None,
        gen_kwargs={},
        parser_kwargs={},
        batch_kwargs={},
    ) -> None:
        super().__init__(block_name)
        self.block_config = self._load_config(config_path)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.client = client
        if trans_model_id:
            self.trans_model_id = trans_model_id
        else:
            # get the default model id from client
            self.trans_model_id = self.client.models.list().data[0].id
        self.output_cols = output_cols
        self.batch_params = batch_kwargs
        self.parser_name = parser_kwargs.get("parser_name", None)
        self.parsing_pattern = parser_kwargs.get("parsing_pattern", None)
        self.parser_cleanup_tags = parser_kwargs.get("parser_cleanup_tags", None)
        self.defaults = {
            "temperature": 0,
            "max_tokens": 4096,
        }

        # Whether the LLM server supports a list of input prompts
        # and supports the n parameter to generate n outputs per input
        self.server_supports_batched = False

    def _translate(self, text: str) -> str:
        """Translates a single string and returns the translated text."""
        logging.debug(f"Translating text using model {self.trans_model_id}")

        response = self.client.completions.create(
            model=self.trans_model_id,
            prompt=text,
            extra_body={
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "max_length": 512,
            },
        )

        return response.choices[0].text

    def _translate_samples(self, samples) -> list:
        logger.debug(f"Starting translation...:")

        results = []
        progress_bar = tqdm(range(len(samples)), desc=f"{self.block_name} Translation")
        for sample in samples:

            columns_to_translate = [sample[key] for key in self.block_config.keys()]

            translated_texts = []

            for text in columns_to_translate:
                translated_texts.append(self._translate(text))

            results.append(translated_texts)
            progress_bar.update(1)
        return results

    def generate(self, samples: Dataset) -> Dataset:
        """
        Generate the output from the block. This method should first validate the input data,
        then generate the output, and finally parse the generated output before returning it.
        Args:
            samples (Dataset): The samples used as input data
        Returns:
            The parsed output after generation.
        """

        num_samples = self.batch_params.get("num_samples", None)
        logger.debug("Generating outputs for {} samples".format(len(samples)))

        if (num_samples is not None) and ("num_samples" not in samples.column_names):
            samples = samples.add_column("num_samples", [num_samples] * len(samples))

        # validate each sample
        # Log errors and remove invalid samples
        valid_samples = []

        for sample in samples:
            is_valid = True
            for key in self.block_config.keys():
                if key not in sample:
                    is_valid = False

            if is_valid:
                valid_samples.append(sample)

        samples = valid_samples

        if len(samples) == 0:
            return Dataset.from_list([])

        # generate the output

        outputs = self._translate_samples(samples)

        num_parallel_samples = 1
        extended_samples = []

        # Duplicate each input sample n times, where n is the number
        # of output sequences generated per input, so that we can
        # pair up the inputs and outputs.
        for item in samples:
            extended_samples.extend([item] * num_parallel_samples)

        new_data = []
        for sample, output in zip(extended_samples, outputs):

            translated_data = {}

            index = 0
            for key in self.output_cols:
                translated_data[key] = output[index]
                index = index + 1

            new_data.append({**sample, **translated_data})

        return Dataset.from_list(new_data)
