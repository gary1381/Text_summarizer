# Text_summarizer

This project implements a text summarization system using state-of-the-art natural language processing techniques.

## Features

- Automatic text summarization
- Fine-tuning of pre-trained language models
- Evaluation metrics for summarization quality
- Support for both extractive and abstractive summarization

## Technologies

- Python 3.x
- PyTorch
- Transformers library (Hugging Face)
- NLTK (Natural Language Toolkit)
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation) for evaluation

## Project Structure

- `train.py`: Main script for training the summarization model



## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/Text_summarizer.git
   cd Text_summarizer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset in the required format.
2. Run the training script:
   ```
   python train.py
   ```

3. For inference, use:
   ```
   python inference.py --input "Your text to summarize"
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

