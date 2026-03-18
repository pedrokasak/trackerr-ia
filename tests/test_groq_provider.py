import pytest
from unittest.mock import MagicMock, patch
from benchmark.providers.groq_provider import GroqProvider

@pytest.fixture
def mock_groq_client():
    with patch("groq.Groq") as mock:
        yield mock

@pytest.mark.asyncio
async def test_groq_provider_analyze(mock_groq_client):
    # Setup
    mock_instance = mock_groq_client.return_value
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content='{"portfolio_assessment": "Bom", "key_insights": []}'))
    ]
    mock_instance.chat.completions.create.return_value = mock_completion
    
    with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
        provider = GroqProvider()
        result = await provider.analyze("teste prompt")
        
        # Assertions
        assert result["portfolio_assessment"] == "Bom"
        assert provider.provider_name == "groq"
        mock_instance.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_groq_provider_parse_error(mock_groq_client):
    mock_instance = mock_groq_client.return_value
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content='Texto sem json'))
    ]
    mock_instance.chat.completions.create.return_value = mock_completion
    
    with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
        provider = GroqProvider()
        result = await provider.analyze("teste prompt")
        
        assert "raw_response" in result
        assert result["raw_response"] == "Texto sem json"
