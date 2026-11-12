"""
Voz do Copiloto (ADR-007): transcrição (STT) e síntese de voz (TTS).

O server chama estes serviços pelos endpoints `/api/speech/*` e não sabe
qual provider ou modelo está em uso. A escolha fica no `SpeechFactory`, por
variável de ambiente, no mesmo padrão do `LLMFactory`.
"""
