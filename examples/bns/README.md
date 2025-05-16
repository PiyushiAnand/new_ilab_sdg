# Synthetic Scenario-Response Generation for Legal Retrieval

This notebook demonstrates how to use the `sdg` package to generate synthetic scenario-response data with Granite 3.3 2B as the teacher model and Facebook NLLB translation model. The generated data is in Kannada and can be used to improve retriever capability in the legal domain as well as legal question-answering systems.

## Table of Contents
- [Overview](#overview)
- [Generation Pipeline Overview](#reasoning-pipeline-overview)
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
    A[Input Section] --> B[Scenario Generation]
    B --> C[Response Generation]
    C --> D[Scenario and Response Tranlation]
```

### Scenario-Response Generation

* Our SDG pipeline leverages the generation capabilities of language models to generate a diverse set of scenrios and responses for the provided sections from Bharatiya Nyaya Sanhita.
* Scenario represents real-life questions which a common person might ask a lawyer.

### Scenario-Response Translation

* Once we generate scenarios and responses, they are translated to Kannada using Facebook NLLB translation model.

## Notebook Structure
The notebook is logically divided in below sections:
### Creating ICLs
- SDG works by creating a set Scenario-Response pairs from the source document.
- To do this we first need to create an example document and a set of Scenario-Response pairs. The SDG will use these to generate more synthetic Scenario-Response pairs on top of all your document.

### Generating Data
- In next few sections you will:
    - Learn how we added prompts for the teacher model
    - Generated data

## How does the generated data look like?

#### Input Raw Document
```text
"Chapter_III

Section_27

not extend to (a) the intentional causing of death, or to the attempting to cause death; (b) the doing of anything which the person doing it knows to be likely to cause
```


### Generated data

#### Generated Scenario-Response Pair
```text
Scenario: Suppose I urge my friend C to steal a car from a dealership, knowing full well it's illegal. Although I don't physically assist in the theft, C goes ahead and takes the vehicle. Can I face the same legal repercussions for abetting this crime under the Bharatiya Nyaya Sanhita?

Response: In accordance with Chapter III, Section 27 of the Bharatiya Nyaya Sanhita, yes, you can be held equally liable as the perpetrator under such circumstances. This principle applies to situations where an individual intentionally abets a crime, including the act of theft in this case, even when they do not physically participate in the commission.
```

#### Translated Scenario-Response Pair
```text
Scenario:


Response:
```


