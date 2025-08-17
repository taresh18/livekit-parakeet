import asyncio
import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import requests
import time
from requests.adapters import HTTPAdapter

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.utils import AudioBuffer
from livekit.agents.stt import SpeechEvent, SpeechEventType

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 24000
ASR_SAMPLE_RATE = 16000


@dataclass
class CanaryOptions:
    """Configuration options for CanarySTT."""
    server_url: str
    language: str

class CanarySTT(stt.STT):
    """STT implementation using NVIDIA Nemo Parakeet"""
    
    def __init__(
        self,
        server_url: str = "http://localhost:8989",
        language: str = 'en',
    ):
        """Initialize the CanarySTT instance.
        
        Args:
            server_url: URL of the FastAPI parakeet service
            language: Language code for speech recognition
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        self._opts = CanaryOptions(
            server_url=server_url,
            language=language,
        )
                
        # Create session for reuse
        self._session = requests.Session()
        
        # Configure connection pooling for localhost
        adapter = HTTPAdapter(
            max_retries=0,              # No retries for speed
            pool_connections=10,        # More connections available
            pool_maxsize=20,           # Larger pool for reuse
            pool_block=False           # Don't block on pool exhaustion
        )
        
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # Aggressive keep-alive configuration for persistent connections
        self._session.headers.update({
            'User-Agent': 'CanarySTT/1.0',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=300, max=10000',
        })
        
        # Configure session-level connection pooling
        self._session.trust_env = False
        
        # Pre-create optimized headers template
        self._base_headers = {
            'Content-Type': 'application/octet-stream',
            'Accept': 'application/json'
        }

    def update_options(
        self,
        *,
        server_url: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """Update STT options.
        
        Args:
            server_url: FastAPI server URL
            language: Language code for speech recognition
        """
        if server_url:
            self._opts.server_url = server_url
        if language:
            self._opts.language = language

    def _sanitize_options(self, *, language: Optional[str] = None) -> CanaryOptions:
        """Create a copy of options with optional overrides.
        
        Args:
            language: Language override
            
        Returns:
            Copy of options with overrides applied
        """
        options = dataclasses.replace(self._opts)
        if language is not None:
            options.language = language
        return options

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: Optional[str],
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Implement speech recognition via HTTP service.
        
        Args:
            buffer: Audio buffer
            language: Language to detect
            conn_options: Connection options
            
        Returns:
            Speech recognition event
        """
        error_msg = None
        
        try:
            options = self._sanitize_options(language=language)
            
            # Extract raw PCM data from buffer
            wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
            raw_pcm_data = wav_bytes[44:]  # Remove WAV header 
            
            # Minimal headers
            headers = {'Content-Type': 'application/octet-stream'}
            params = {'sample_rate': AUDIO_SAMPLE_RATE}
            
            # Record start time for performance logging
            start_time = time.time()
            
            response = self._session.post(
                f"{options.server_url}/v1/transcribe/canary",
                data=raw_pcm_data,
                headers=headers,
                params=params,
                timeout=10.0,
            )
            
            processing_time = time.time() - start_time
                            
            if response.status_code == 200:
                result = response.json()
                full_text = result.get('text', '').strip()
                server_processing_time = result.get('processing_time', 0)
                audio_duration = result.get('audio_duration', 0)
                logger.info(f"HTTP transcription successful: {processing_time*1000:.1f}ms total, {server_processing_time*1000:.1f}ms processing, {audio_duration:.3f}s audio, transcript: {full_text}")
            else:
                error_msg = f"HTTP transcription failed with status {response.status_code}: {response.text}"
                raise Exception(error_msg)
                
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=full_text or "",
                        language=options.language,
                    )
                ],
            )

        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP request error: {str(e)}"
            logger.error(f"HTTP request error in speech recognition: {e}", exc_info=True)
            raise APIConnectionError() from e
        except Exception as e:
            error_msg = f"General error: {str(e)}"
            logger.error(f"Error in speech recognition: {e}", exc_info=True)
            raise APIConnectionError() from e

    def __del__(self):
        """Clean up resources properly."""
        try:
            if hasattr(self, '_session') and self._session:
                self._session.close()
                logger.debug("Session closed in destructor")
        except Exception as e:
            logger.debug(f"Error closing session in destructor: {e}")
