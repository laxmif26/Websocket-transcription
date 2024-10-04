async def transcribe_stream(
    ws: WebSocket,
    model: Annotated[ModelName, Query()] = config.whisper.model,
    language: Annotated[Language | None, Query()] = config.default_language,
    response_format: Annotated[ResponseFormat, Query()] = config.default_response_format,
    temperature: Annotated[float, Query()] = 0.0,
) -> None:
    await ws.accept()
    
    # Define transcription options
    transcribe_opts = {
        "language": language,
        "temperature": temperature,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }

    # Load the model and initialize ASR
    whisper = load_model(model)
    asr = FasterWhisperASR(whisper, **transcribe_opts)
    audio_stream = AudioStream()

    # Use a single transcription flag to prevent duplicates
    transcription_sent = False

    async for transcription in audio_transcriber(asr, audio_stream):
        if transcription_sent:
            break  # Exit loop if transcription has already been sent

        logger.debug(f"Sending transcription: {transcription.text}")

        # Send the response based on the requested format
        if ws.client_state == WebSocketState.DISCONNECTED:
            break
        
        if response_format == ResponseFormat.TEXT:
            await ws.send_text(transcription.text)
        elif response_format == ResponseFormat.JSON:
            await ws.send_json(TranscriptionJsonResponse(text=transcription.text).model_dump())
        elif response_format == ResponseFormat.VERBOSE_JSON:
            await ws.send_json(TranscriptionVerboseJsonResponse(text=transcription.text).model_dump())

        transcription_sent = True  # Set the flag after sending the transcription

    logger.info("Closing the connection.")
    await ws.close()
