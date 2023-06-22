# DataWiz &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://reactjs.org/docs/how-to-contribute.html#your-first-pull-request)

![DataWiz](./media/DataWiz.png)

DataWiz is a chat application that allows you to directly retrieve answers from Document files by simply chatting with it.

This open-source project empowers users to extract information from documents in a conversational manner, providing a user-friendly and efficient way to access relevant content.

Currently DataWiz supports .txt, .pdf, .docx, .xlsx, .csv, webpages and youtube videos. More formats will be added soon. It not only extracts the text but also the tables and images from the documents. While preserving the context.

## Features

- **Interactive Chat**: Engage in natural language conversations with DataWiz to obtain specific information from PDF files.
- **Local Data Processing**: All data processing occurs locally on your computer, ensuring privacy and security. No data is sent outside your machine during the chat process.
- **Versatile Deployment**: Run DataWiz on your local machine or virtual machines like AWS to suit your preferred environment.
- **MIT License**: DataWiz is released under the permissive MIT License, allowing you to use, modify, and distribute the software with minimal restrictions.

## Technology Stack

DataWiz utilizes the following technologies:

- **LangChain**: A framework for developing applications powered by language models.
- **StableVicuna-13B**: Open Soure Large Language Model (LLM) that runs locally on your preferred machine.
- **FAISS**: FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other.
- **Hugging Face Sentence Transformers**: `all-MiniLM-L6-v2` sentence transformer model is used in this project. It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

## Installation

To run DataWiz locally, follow these steps:

1. Clone the repository
2. Install the necessary dependencies:
3. Ingest the data: `python ingest.py`
4. Run the application: `python main.py`

For deploying DataWiz on AWS or other virtual machines, refer to the respective documentation for detailed instructions.

## Usage

Once DataWiz is up and running, access the application via your command-line interface. Engage in conversations by typing queries, and DataWiz will provide answers based on the content extracted from the provided files.

## Contributing

Contributions to DataWiz are welcome! If you encounter any issues, have suggestions, or would like to contribute code improvements, please submit a pull request or open an issue in the GitHub repository.

Before making contributions, please review our [contribution guidelines](CONTRIBUTING.md) to ensure a smooth collaboration process.

## License

DataWiz is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the software under the terms of this license.

## Acknowledgements

We would like to express our gratitude to the developers of the libraries and frameworks that made this project possible. Special thanks to the creators of LangChain, Vacunia-13, and Hugging Face Embeddings for their invaluable contributions to the open-source community.

## Contact

For any questions or inquiries about DataWiz, feel free to reach out at [brohan5501@gmail.com](brohan5501@gmail.com). I appreciate your feedback and suggestions to enhance the application further.

---

Thank you for using DataWiz! We hope this chat application simplifies your document content extraction process and improves your productivity.
