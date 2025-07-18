# Industry-Specific AI Agents

This repository is a collection of practical, tool-using AI agents designed to solve specific problems in various industries. Each project is a self-contained exploration of different agentic architectures and capabilities, built using Python and powered by OpenAI's GPT models.

## Core Concepts Explored

The projects in this repository demonstrate several key concepts in modern AI agent design:

* **Tool-Using Agents:** The core of each agent is an LLM that can intelligently choose from and use a "tool belt" of custom Python functions to perform actions or retrieve information.
* **Retrieval-Augmented Generation (RAG):** Agents are grounded in specific knowledge bases (e.g., FAQ documents) to provide accurate, factual answers and prevent hallucination.
* **Hybrid Architecture:** This repository champions a robust hybrid model that combines the flexible, language-understanding power of an AI agent with the reliability of programmatic, rules-based code for critical business processes.

---

## Projects in This Repository

### 1. Consumer Banking Agent

A conversational AI assistant designed to handle common customer service inquiries for a consumer bank. This agent acts as the first line of support, capable of answering questions and handling escalations.

* **Location:** [`./Consumer Banking/`](./Consumer%20Banking/)
* **Architecture:** Implemented using the **Hybrid Agent Model**.
* **Key Features:**
    * **Intelligent FAQ Handling:** Uses a RAG pipeline to understand and answer customer questions from a dedicated knowledge base.
    * **Agentic Escalation:** Autonomously infers the urgency of a user's problem (e.g., a blocked account) and uses a tool to create a support case.
    * **Programmatic Closing Ceremony:** Once a user is finished, the agent transitions to a reliable, scripted closing sequence to present offers, provide a summary, and collect feedback.

*(More projects will be added here in the future...)*

---

## Setup and Usage

### Prerequisites

* Python 3.9+
* An OpenAI API Key

### Installation & Configuration

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/lvjr3383/Industry-Agents.git](https://github.com/lvjr3383/Industry-Agents.git)
    cd Industry-Agents
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *(This assumes a `requirements.txt` file is present)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your API Key:**
    * Create a file named `.env` in the root of the `AGENTS` project directory.
    * Add your OpenAI API key to this file:
        ```
        OPENAI_API_KEY="sk-..."
        ```

### Running an Agent

Each agent is a self-contained project. To run an agent, navigate to its directory and execute its main script.

**Example for the Consumer Banking Agent:**

```bash
cd "Consumer Banking"
python3 hybrid.py