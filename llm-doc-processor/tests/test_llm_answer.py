
import unittest
from unittest.mock import MagicMock, patch
from llm_answer import DecisionEngine, ParsedQuery, EmbeddedChunk

class TestDecisionEngine(unittest.TestCase):

    @patch('llm_answer.genai.GenerativeModel')
    def test_make_decision_limits_chunks(self, mock_generative_model):
        # Arrange
        mock_model_instance = MagicMock()
        mock_generative_model.return_value = mock_model_instance
        mock_model_instance.generate_content.return_value.text = '{"decision": "Approved"}'

        decision_engine = DecisionEngine()
        mock_query = ParsedQuery(
            original_query="Is my knee surgery covered?",
            enhanced_query="knee surgery coverage policy claim",
            entities={},
            intent='claim_inquiry',
            keywords=['knee', 'surgery', 'coverage'],
            confidence=0.9
        )
        
        # Create more chunks than the default limit (5)
        mock_chunks = [
            (EmbeddedChunk(chunk=MagicMock(), embedding=None, embedding_model='', embedding_dim=0), 0.95),
            (EmbeddedChunk(chunk=MagicMock(), embedding=None, embedding_model='', embedding_dim=0), 0.94),
            (EmbeddedChunk(chunk=MagicMock(), embedding=None, embedding_model='', embedding_dim=0), 0.93),
            (EmbeddedChunk(chunk=MagicMock(), embedding=None, embedding_model='', embedding_dim=0), 0.92),
            (EmbeddedChunk(chunk=MagicMock(), embedding=None, embedding_model='', embedding_dim=0), 0.91),
            (EmbeddedChunk(chunk=MagicMock(), embedding=None, embedding_model='', embedding_dim=0), 0.90)
        ]

        # Act
        decision_engine.make_decision(mock_query, mock_chunks)

        # Assert
        # Check that the prompt sent to the LLM only contains the top 5 chunks
        call_args, call_kwargs = mock_model_instance.generate_content.call_args
        prompt = call_args[0]
        
        self.assertIn("--- Chunk from", prompt)
        self.assertEqual(prompt.count("--- Chunk from"), 5)

if __name__ == '__main__':
    unittest.main()
