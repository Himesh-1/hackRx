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
            (EmbeddedChunk(chunk=MagicMock(content=f"Mock content {i}", source_document=f"doc{i}.pdf"), embedding=None, embedding_model='', embedding_dim=0), 0.95 - i*0.01)
            for i in range(6)
        ]

        # Act
        processed_questions = [
            {
                "question": mock_query.original_query,
                "context": [item[0] for item in mock_chunks] # Pass the chunk object
            }
        ]
        decision_engine.make_decisions_batch(processed_questions)

        # Assert
        # Check that the prompt sent to the LLM contains the expected format and number of chunks
        call_args, call_kwargs = mock_model_instance.generate_content.call_args
        prompt = call_args[0]
        
        self.assertIn("[Clause 1] Source: doc0.pdf", prompt)
        self.assertEqual(prompt.count("[Clause"), 6) # Expecting 6 clauses to be included

        # Verify that the LLM was called with the correct prompt
        mock_model_instance.generate_content.assert_called_once()

if __name__ == '__main__':
    unittest.main()