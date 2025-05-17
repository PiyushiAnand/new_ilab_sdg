# Synthetic Question-Answer Generation over Wikipedia Documents

This notebook demonstrates how to use the `sdg` package to generate synthetic question-answer pairs using Kannada Wikipedia with Granite 3.3 2B as the teacher model. Since Granite 3.3 2B doesn't support Kannada we translate the wikipedia documents to English, generate question-answer pairs in English and translated the generated question-answer pairs back to Kannada. We will use [IndicTrans v2 translation model](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface) for the same. 

## Table of Contents
- [Overview](#overview)
- [Generation Pipeline Overview](#generation-pipeline-overview)
- [Notebook Structure](#notebook-structure)
  - [Creating ICLs](#creating-icls)
  - [Generating Data](#generating-data)

## Overview

The workflow includes:

- Defining an SDG pipeline using a YAML flow file
- Creating custom SDG blocks for providing translation supprot

## Generation Pipeline Overview

```mermaid
graph LR
    A[Input Wikipedia Passage] --> B[Kannada Passage Translation to English]
    B --> C[Question Generation]
    C --> D[Answer Generation]
    D --> E[Question and Answer Translation to Kannada]
```

### Kannada Passage Translation to English
We use IndicTrans v2, specifically `ai4bharat/indictrans2-indic-en-dist-200M` model to translate Kannada Wikipedia passages to English.

### Question Generation

* Our SDG pipeline leverages the generation capabilities of language models to generate a diverse set of question and answers based on the translated passages.

### Answer Generation

* Once we generate questions, we leverages the generation capabilities of language models to generate answer to the question grounded on the document.

### Question and Answer Translation to Kannada
We use IndicTrans v2 `ai4bharat/indictrans2-en-indic-dist-200M` model to translate generated question-answer pairs back to Kannada.

## How does the generated data look like?

#### Input Raw Document
```text
ಎಸ್.ನಿಜಲಿಂಗಪ್ಪ
ಕರ್ನಾಟಕದ ಏಕೀಕರಣಕ್ಕೆ ಹೋರಾಡಿದ ರಾಜಕಾರಣಿಗಳಲ್ಲಿ ಪ್ರಮುಖರು
```

#### Translated Document in English
```text
S. Nijalingappa
Prominent among the politicians who fought for the unification of Karnataka
```

#### Generated Question-Answer Pair
```text
Question: Who was a prominent politician known for fighting for the unification of Karnataka, as mentioned in the document?

Answer: S. Nijalingappa
```

#### Translated Question-Answer Pair
```text
Question: 
ಡಾಕ್ಯುಮೆಂಟ್ನಲ್ಲಿ ಉಲ್ಲೇಖಿಸಿರುವಂತೆ, ಕರ್ನಾಟಕದ ಏಕೀಕರಣಕ್ಕಾಗಿ ಹೋರಾಡಿದ ಪ್ರಸಿದ್ಧ ರಾಜಕಾರಣಿ ಯಾರು?

Answer:
ಎಸ್.ನಿಜಲಿಂಪ್ಪ
```

