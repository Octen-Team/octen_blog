---
layout: post
title: "Octen Series: Optimizing Embedding Models to #1 on RTEB Leaderboard"
date: 2026-01-13
categories: [AI, Embedding, RTEB]
author: Octen Team
---

# Octen Series: Optimizing Embedding Models to #1 on RTEB Leaderboard

## 1. Background and Motivation

### 1.1 RTEB: The True Litmus Test for Retrieval Capability

**[RTEB (Retrieval Embedding Benchmark)](https://huggingface.co/blog/rteb)** is a new generation of retrieval evaluation benchmark launched by MTEB. Compared to traditional MTEB, RTEB focuses on **real-world industry application scenarios** and aims to address the "overfitting" problem of models on public benchmarks.

**Core Features of RTEB**:

1. **Industry-Oriented**: Covers key enterprise domains
   - **Legal**: Legal document retrieval in German, English, Japanese, and French
   - **Finance**: Financial reports, Q&A, and personal finance content
   - **Healthcare**: Medical Q&A, clinical dialogues, and health consultations
   - **Code**: Programming problems, code search, and SQL queries

2. **Hybrid Evaluation Strategy**: Open datasets + Private datasets
   - **Open Datasets**: Fully public corpus and labels, reproducible results
   - **Private Datasets**: Managed by MTEB maintainers to prevent models from memorizing test data
   - The **performance gap** between the two directly reflects the model's true generalization capability

3. **Full Multilingual Coverage**: Supports 20 languages including Chinese, English, German, Japanese, French, etc.

4. **Evaluation Metric**: Uses **NDCG@10** as the core metric to measure retrieval ranking quality

### 1.2 Challenges in Industry-Specific Vertical Domain Retrieval

Compared to general text retrieval, industry vertical scenarios face more severe challenges:

1. **Domain Knowledge Barriers**
   - Legal terminology and statutory citations in the legal domain
   - Understanding of financial figures and metrics in the finance domain
   - Diseases, medications, and treatment procedures in the healthcare domain
   - Programming language syntax and API understanding in the code domain

2. **Ultra-Long Document Processing**
   - Legal documents can reach tens of thousands of words
   - Medical records contain complex patient histories and examination reports
   - Code files involve multi-file dependencies

3. **Data Scarcity and Diversity**
   - Difficulty in obtaining high-quality annotated data
   - Real queries contain noise such as spelling errors and synonym variations
   - Diverse query intents (exact matching, semantic understanding, cross-lingual retrieval)

4. **High Generalization Requirements**
   - Models need to perform well on unseen domain data
   - Avoid overfitting to public test sets and ensure practical application effectiveness

**These challenges motivated us to design a systematic optimization approach, ultimately enabling the Octen 8B model to achieve first place on the [RTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).**

![RTEB Leaderboard Results]({{ site.baseurl }}/assets/images/posts/octen_rteb_leaderboard.png)
*Figure 1: Octen-Embedding-8B achieves first place on the RTEB leaderboard*

---

## 2. Core Technical Optimizations

![Octen Embedding Model Overview]({{ site.baseurl }}/assets/images/posts/octen_embedding.png)
*Figure 2: Overview of Octen embedding model architecture and training pipeline*

### 2.1 Synthetic Data Optimization

High-quality training data is the cornerstone of model performance. For the four major domains covered by RTEB, we designed domain-differentiated synthetic data strategies.

#### 2.1.1 Domain-Specific Data Sources

We adopted differentiated data construction strategies for different domain characteristics:

**Legal Domain**
- **Data Sources**: Legal documents, case law, statutes, contract templates
- **Characteristics**: Ultra-long documents (up to tens of thousands of words), dense technical terminology, complex citation relationships
- **Synthesis Strategy**: Extract key clauses from legal documents to generate queries like "find relevant case law" and "clause interpretation"

**Finance Domain**
- **Data Sources**: Financial reports, finance Q&A, investment analysis, personal finance consulting
- **Characteristics**: Numerical sensitivity, metric understanding, temporal relationships
- **Synthesis Strategy**: Generate queries like "company financial status analysis" and "industry comparison" based on key financial metrics

**Healthcare Domain**
- **Data Sources**: Medical Q&A, clinical dialogues, disease encyclopedias, medication guides
- **Characteristics**: Diverse symptom descriptions, disease-treatment associations, doctor-patient dialogue features
- **Synthesis Strategy**: Generate queries like "diagnostic recommendations" and "medication consultation" from symptom descriptions

**Code Domain**
- **Data Sources**: Programming Q&A, code documentation, API documentation, SQL queries
- **Characteristics**: Multiple programming languages, syntax structures, API calls
- **Synthesis Strategy**: Generate queries like "implement functionality," "debug errors," and "SQL query construction" from code snippets

#### 2.1.2 Query Diversity Enhancement

In real retrieval scenarios, user queries come in various forms. We enhance query diversity through the following strategies:

**One-to-Many**: One query corresponds to multiple relevant documents
- Scenario: Users search for a topic expecting multiple related cases or solutions
- Example: Query "Python exception handling" may correspond to multiple different exception handling tutorials

**Many-to-One**: Multiple queries correspond to the same document
- Scenario: The same document can answer questions from different perspectives
- Example: A medical article can answer multiple queries like "what are the symptoms," "how to treat," "prevention measures"

**Noise Injection**: Simulating real user input
- **Spelling Errors**: Introduce common spelling mistakes (e.g., "retreival" → "retrieval")
- **Synonym Substitution**: Use synonymous expressions within the domain (e.g., "find" → "retrieve" → "search")
- **Expression Variation**: Change query sentence structures (declarative sentences, questions, keyword combinations)

This diversity enhancement strategy enables the model to adapt to various forms of user queries in real scenarios.

### 2.2 Data Optimization Strategies

In addition to data synthesis, we conducted deep optimization in the organization and utilization of training data.

#### 2.2.1 Hard Negative Sampling

**Core Idea**: Randomly sampled negative samples are often too simple and easy for models to distinguish, leading to limited training effectiveness. We introduced a Hard Negative sampling mechanism to select documents that are semantically similar to queries but actually irrelevant as negative samples.

**Optimization Directions**:
- Leverage existing models' ranking capabilities to identify hard negatives
- Ensure sampled negative samples are challenging in quality
- Avoid introducing false negatives that are actually positive samples

**Effect**: Through hard sample training, models are forced to learn finer-grained semantic distinctions, significantly improving discrimination accuracy in real retrieval scenarios.

#### 2.2.2 Multi-Positive Full Utilization

**Problem Identification**: Many datasets annotate multiple positive documents for each query (averaging 4-8), but traditional training methods cannot fully utilize these annotations, leading to waste of large amounts of high-quality data (waste rate up to 70-90%).

**Optimization Strategy**: We designed a data utilization mechanism to ensure all annotated positive pairs are fully learned during training.

**Effects**:
- Data utilization rate increased to nearly 100%
- Increased training steps for more thorough model convergence
- Fully exploited the value of annotated data

#### 2.2.3 False Negative Filtering

**Core Problem**: In contrastive learning, randomly sampled negative samples may actually be semantically related positive samples (false negatives), providing incorrect supervision signals to the model.

**Optimization Approach**: We developed a dynamic filtering mechanism to identify and filter false negatives in real-time during training, ensuring training signal accuracy.

**Effects**:
- Reduced training noise and improved model discrimination accuracy
- Particularly effective for datasets with smaller corpus (higher probability of false negatives)

### 2.3 Training Efficiency Improvements

To support large-scale, multi-domain training, we conducted systematic optimization of training efficiency.

#### 2.3.1 LoRA Fine-tuning Based on Qwen3

**Model Selection**: We chose the Qwen3 series (8B/4B/0.6B) as base models due to their advantages in multilingual understanding and long-text processing.

**LoRA Fine-tuning Strategy**:
- Adopt parameter-efficient LoRA (Low-Rank Adaptation) method
- Only fine-tune query/key/value projection layers and feedforward network layers in the model
- Significantly reduce trainable parameters (from billions to tens of millions)
- Save memory and training time while maintaining model performance

**Advantages**:
- Training speed improved 2-3x
- Memory usage reduced by over 50%
- Can train multiple LoRA adapters for different domains
- Small model checkpoint size (only tens of MB), convenient for management and deployment

#### 2.3.2 Cross-Device Negative Sharing

**Core Idea**: In multi-GPU training, globally share query and document embeddings from all devices, allowing each GPU to see global negatives.

**Workflow**:
1. Each GPU encodes queries and documents in its own batch
2. Collect embeddings from all GPUs via distributed communication mechanism (all_gather)
3. Calculate global contrastive loss, effective batch size = single GPU batch size × number of GPUs
4. Expand negative pool and improve contrastive learning effectiveness

**Effects**:
- In 8-card training, effective negative count increased 8x
- Significantly improved contrastive learning training quality
- No increase in single-card memory pressure

#### 2.3.3 Adaptive Batch Size Optimization

**Key Finding**: The bottleneck for batch size is mainly in corpus encoding (each batch needs to encode batch_size × (1 + neg_per_ins) documents), not query encoding.

**Optimization Strategy**:
- Dynamically adjust batch size based on sequence length
- Short sequences (<500 tokens): Use larger batch size (32-64)
- Medium sequences (500-3000 tokens): Moderate batch size (4-20)
- Ultra-long sequences (>15000 tokens): Small batch size (1-4)

**Effects**:
- Fully utilize GPU memory
- Avoid OOM (Out of Memory)
- Improve training throughput

### 2.4 Industry Scenario Adaptation

We conducted scenario-specific adaptation optimizations for different industry needs.

#### 2.4.1 Multi-Domain Separate Training

**Strategy**: Train domain-specific models separately for the four major domains: legal, finance, healthcare, and code.

**Advantages**:
- Each model focuses on semantic features of a specific domain
- Avoid "averaging" effects from mixed multi-domain training
- Domain models can use more training steps and finer-grained parameter tuning

#### 2.4.2 Ultra-Long Sequence Processing

**Challenge**: Legal and healthcare domains contain many ultra-long documents (10K-50K tokens).

**Solutions**:
- Support variable maximum length configuration (set based on dataset characteristics)
- Use Gradient Checkpointing to reduce memory usage
- Use small batch size (even batch=1) for ultra-long sequence datasets
- Maintain effective batch size through gradient accumulation

#### 2.4.3 Model Fusion

**Core Idea**: Fuse the weights of multiple domain-specific models to obtain a general-purpose model with stronger comprehensive performance.

**Fusion Strategy**:
- Perform weighted averaging of parameters from multiple domain models
- Weights can be adjusted based on data volume or importance of each domain
- Fused model retains professional capabilities in each domain while having cross-domain generalization ability

**Effects**:
- Octen 8B model achieved first place on the overall RTEB leaderboard through multi-domain fusion
- Maintained excellent performance across all sub-domains
- Strong generalization ability, suitable for general retrieval scenarios

---

**Summary**: Through systematic optimization in these four dimensions, we built a complete training pipeline from data synthesis, data utilization, training efficiency to scenario adaptation, forming the core technical advantages of the Octen series models.

## 3. Challenges and Responses: Breaking Through in Unfair Competition

### 3.1 The Unfairness of RTEB Private Datasets

In our pursuit of first place on the RTEB leaderboard, we discovered and publicly discussed a structural issue that severely affected evaluation fairness.

#### 3.1.1 Problem Discovery

RTEB's original design uses a "open dataset + private dataset" hybrid strategy to detect model generalization capability and prevent overfitting. However, we pointed out in [GitHub issue #3902](https://github.com/embeddings-benchmark/mteb/issues/3902): **the so-called "private" datasets are not confidential to all participants**.

**Core Unfairness Points**:
1. **Structural Access Privilege Disparity**: Some participants have access to private datasets while others cannot obtain them
2. **Implicit Optimization Risk**: Participants with access to evaluation tasks can optimize models in a targeted manner, creating unfair advantages even unintentionally
3. **Empirical Evidence**: Through analysis of leaderboard data, some models' main advantages are concentrated in "private" datasets rather than public benchmarks

#### 3.1.2 Community Response and Our Call to Action

Our concerns were acknowledged by the MTEB team and community:
- MTEB core members acknowledged the structural advantage problem
- Community began discussing temporarily removing private track influence on the leaderboard
- Proposed establishing a multi-party oversight committee

**The Essence of This Problem**: An evaluation mechanism designed to prevent overfitting has created new unfairness due to unequal data access rights.

**Our Call to Action**:

We hope to see a more open and fair evaluation environment. To this end, we call for:

1. **More participants to contribute high-quality private datasets**: Encourage teams from different industries and scenarios to share real business problems and datasets
2. **Introduction of real business scenarios**: Bring actual retrieval challenges faced by enterprises into the evaluation system, making benchmarks more aligned with practical applications
3. **Open sharing mechanism**: Establish transparent dataset contribution and management mechanisms to ensure all participants compete under equal conditions
4. **Promote domain development**: Drive progress of retrieval models in various vertical domains through diversified high-quality datasets

Only evaluations built on fair and open foundations can truly promote the healthy development of embedding and retrieval technologies.

### 3.2 Breaking Through with Technical Excellence

Facing such an unfair competitive environment, we chose **not to complain, but to prove our strength through technical breakthroughs**.

#### 3.2.1 Large-Scale Domain Synthetic Data

Since we couldn't access private datasets, we built larger-scale, higher-quality domain data ourselves:

**Scale Enhancement**:
- Built large-scale synthetic training data for the four major domains: legal, finance, healthcare, and code
- Covered all 20 languages and vertical domains involved in RTEB
- Data scale reached millions of query-document pairs

**Quality-First Strategy**:
- In-depth study of each domain's characteristics and real retrieval scenarios
- Introduced noise data to simulate real user queries
- Improved data quality through Hard Negative sampling

**Diversity Assurance**:
- One-to-many and many-to-one query-document relationships
- Comprehensive coverage across multiple languages, domains, and scenarios
- Real noise such as spelling errors, synonyms, and expression variations

#### 3.2.2 Breakthrough: Achieving First Place Through Open and Transparent Methods

**Final Achievement**: Through large-scale domain synthetic data and systematic technical optimization (see Chapter 2), the Octen 8B model achieved **first place** on the RTEB leaderboard.

**The Significance of This First Place**:
- **Validation of Technical Approach**: Proved that through systematic technical optimization and high-quality synthetic data, it's entirely possible to reach SOTA level through open and transparent methods
- **Adherence to Open Spirit**: All our optimizations are based on publicly available data and techniques, without relying on any privileged access
- **True Generalization Capability**: Maintained excellent performance on both open and private datasets, proving the model's genuine generalization ability

**Our Belief**: Technical competition should be built on a fair foundation. True technological progress comes from open, transparent, and reproducible research. The success of the Octen series models demonstrates that adhering to these principles can still achieve excellent results.

---

## 4. Experimental Results

### 4.1 Overall Performance on RTEB Leaderboard

The Octen series models achieved outstanding results on the RTEB leaderboard, with the Octen-Embedding-8B model ranking **first**. The table below shows the comprehensive performance of mainstream embedding models on RTEB:

| Model | Embedding Dimensions | Max Tokens | Mean (Public) | Mean (Private) | Mean (Task) |
|-------|---------------------|------------|---------------|----------------|-------------|
| **Octen-Embedding-8B** | **4096** | **32768** | **0.7953** | **0.8157** | **0.8045** |
| voyage-3-large | 1024 | 32000 | 0.7434 | 0.8277 | 0.7812 |
| gemini-embedding-001 | 3072 | 2048 | 0.7218 | 0.8075 | 0.7602 |
| **Octen-Embedding-4B** | **2560** | **32768** | **0.7747** | **0.7942** | **0.7834** |
| MoD-Embedding | 2560 | 32768 | 0.7642 | 0.7900 | 0.7758 |
| Qwen3-Embedding-8B | 4096 | 32768 | 0.7310 | 0.7838 | 0.7547 |
| voyage-3.5 | 1024 | 32000 | 0.7139 | 0.8102 | 0.7571 |
| Cohere-embed-v4.0 | 1536 | 128000 | 0.6534 | 0.7943 | 0.7166 |
| jina-embeddings-v4 | 2048 | 32768 | 0.6652 | 0.7664 | 0.7105 |
| GritLM-7B | 4096 | 32768 | 0.6187 | 0.7385 | 0.6724 |
| text-embedding-3-large | 3072 | 8191 | 0.6110 | 0.7130 | 0.6567 |
| e5-mistral-7b-instruct | 4096 | 32768 | 0.5090 | 0.7091 | 0.5987 |
| NV-Embed-v2 | 4096 | 32768 | 0.5805 | 0.6691 | 0.6203 |
| snowflake-arctic-embed-l-v2.0 | 1024 | 8192 | 0.5395 | 0.7079 | 0.6150 |
| multilingual-e5-large-instruct | 1024 | 514 | 0.5478 | 0.6859 | 0.6097 |
| gte-multilingual-base | 768 | 8192 | 0.5291 | 0.6697 | 0.5921 |
| text-embedding-005 | 768 | 2048 | - | 0.6294 | - |
| text-embedding-3-small | 1536 | 8191 | 0.5260 | 0.6630 | 0.5874 |
| bge-m3 | 1024 | 8194 | 0.5216 | 0.6726 | 0.5893 |
| Qwen3-Embedding-4B | 2560 | 32768 | - | 0.7711 | - |
| **Octen-Embedding-0.6B** | **1024** | **32768** | **0.7241** | **-** | **-** |
| Qwen3-Embedding-0.6B | 1024 | 32768 | - | 0.7117 | - |

### 4.2 Key Highlights

**Octen-Embedding-8B**:
- **Overall First**: Mean (Task) reaches 0.8045, ranking first among all models
- **Excellent Public Dataset Performance**: 0.7953, demonstrating the model's strong capabilities on publicly visible data
- **Leading Private Dataset Performance**: 0.8157, showcasing outstanding generalization and adaptation to unseen data
- **Ultra-Long Context Support**: Supports 32,768 tokens, suitable for processing long documents in legal, healthcare and other domains

**Octen-Embedding-4B**:
- **Best in 4B Category**: Best performance among models of similar parameter size
- **Balanced Performance**: Maintains high scores on both Public (0.7747) and Private (0.7942) datasets
- **Efficiency Advantage**: Compared to the 8B model, faster inference and lower resource consumption

**Technical Advantages of Octen Series**:
1. **Public-Private Performance Balance**: Similar scores across both dataset types, proving the model doesn't overfit public data and has true generalization capability
2. **Long Context Capability**: 32,768 token support far exceeds most competitors, meeting industry application requirements
3. **High-Dimensional Representation**: 4096/2560-dimensional embedding space provides richer semantic representation capabilities

### 4.3 Multi-Size Model Family

The Octen series provides three different-sized models to meet various scenario requirements:

| Model | Parameters | Use Cases | Characteristics |
|------|-----------|-----------|----------------|
| Octen-Embedding-8B | 7.6B | High-precision retrieval | Best performance, RTEB #1 |
| Octen-Embedding-4B | 4.0B | Balance performance and efficiency | Best in 4B category, faster inference |
| Octen-Embedding-0.6B | 0.6B | Resource-constrained environments | Lightweight deployment, edge devices |

This multi-size model family design allows users to flexibly choose between performance and efficiency based on actual application scenarios.

---

## 5. Open Source Models

The Octen series models have been fully open-sourced on the Hugging Face platform for free use and research by the community.

### 5.1 Open Source Model List

| Model | Hugging Face Link | Parameters | Embedding Dimensions |
|------|------------------|-----------|---------------------|
| Octen-Embedding-8B | [Octen/Octen-Embedding-8B](https://huggingface.co/Octen/Octen-Embedding-8B) | 7.6B | 4096 |
| Octen-Embedding-4B | [Octen/Octen-Embedding-4B](https://huggingface.co/Octen/Octen-Embedding-4B) | 4.0B | 2560 |
| Octen-Embedding-0.6B | [Octen/Octen-Embedding-0.6B](https://huggingface.co/Octen/Octen-Embedding-0.6B) | 0.6B | 1536 |

### 5.2 Usage

**Using Sentence Transformers** (Recommended):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Octen/Octen-Embedding-8B")

# Encode sentences
sentences = [
    "This is an example sentence",
    "Each sentence is converted to a vector"
]

embeddings = model.encode(sentences)
print(embeddings.shape)
# Output: (2, 4096)

# Compute similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.4f}")
```

**Using Transformers**:

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("Octen/Octen-Embedding-8B", padding_side="left")
model = AutoModel.from_pretrained("Octen/Octen-Embedding-8B")
model.eval()

def encode(texts):
    inputs = tokenizer(texts, padding=True, truncation=True,
                      max_length=8192, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # Use last token embedding
        embeddings = outputs.last_hidden_state[:, -1, :]
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

# Example usage
texts = ["Hello world", "你好世界"]
embeddings = encode(texts)
similarity = torch.matmul(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.4f}")
```

### 5.3 Significance of Open Source

**Promoting Technological Progress**: By open-sourcing high-performance embedding models, we hope to:
- Lower technical barriers for industry applications and benefit more developers
- Provide strong baseline models for academic research
- Drive application and innovation of embedding technology in vertical domains

**Community Contribution**: We welcome the community to:
- Test and use Octen models in various application scenarios
- Report issues and provide usage feedback
- Perform secondary development and optimization based on Octen

---

## Summary

The Octen series models achieved first place on the RTEB leaderboard through systematic technical optimization. Our core contributions include:

1. **Domain-Specific Synthetic Data**: Constructed high-quality training data for vertical domains including legal, finance, healthcare, and code
2. **Data Optimization Strategies**: Techniques including Hard Negative sampling, Multi-Positive full utilization, and False Negative filtering
3. **Training Efficiency Optimization**: Methods including LoRA fine-tuning, cross-device negative sharing, and adaptive batch size optimization
4. **Scenario Adaptation Capabilities**: Features including ultra-long document processing, multilingual support, and model fusion

We look forward to the Octen series models bringing value to industry retrieval applications, and we also look forward to more researchers joining open and fair technical competition to collectively promote the development of embedding and retrieval technologies.
