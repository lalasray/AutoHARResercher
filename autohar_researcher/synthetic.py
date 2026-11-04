from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class MCPGenerator:

    servers: dict

    def generate(self, request: dict, length: int, channels: int, seed: int = 13, *, sampling_hz: float = 50) -> np.ndarray:
        kind = request['generator']
        if kind not in self.servers:
            raise ValueError(f'MCP generator {kind!r} is not configured')
        spec = self.servers[kind]
        args = {'label': request['label'], 'count': request['count'],
                'sampling_hz': sampling_hz, 'window_steps': length,
                'channels': channels, 'seed': seed}
        if kind == 'text':
            args['prompt'] = request['prompt']
        else:
            videos = request['videos']
            if not set(videos) <= set(spec.get('allowed_videos', [])):
                raise ValueError('Video not in predefined source video pool')
            args['videos'] = videos
            args['selection'] = request.get('selection', 'all')
        if spec.get('dataset_profiles', False):
            if not request.get('source_dataset') or not request.get('channel_names'):
                raise ValueError('Dataset-aware generation requires source_dataset and channel_names')
            args['dataset'] = request['source_dataset']
            args['channel_names'] = request['channel_names']
        payload = asyncio.run(asyncio.wait_for(self._call(spec, args), timeout=spec.get('timeout_seconds', 600)))
        if kind == 'video':
            provenance = payload.get('provenance') or {}
            if not isinstance(provenance, dict):
                raise ValueError('Video generation provenance must be an object')
            if (payload.get('coordinate_mode') == 'hip-relative'
                    or payload.get('implementation') == 'imutube-inspired-mediapipe'
                    or provenance.get('coordinate_mode') == 'hip-relative'
                    or payload.get('training_eligible') is False
                    or payload.get('physically_complete_imu') is False
                    or payload.get('diagnostic_only') is True
                    or provenance.get('training_eligible') is False
                    or provenance.get('physically_complete_imu') is False
                    or provenance.get('diagnostic_only') is True):
                raise ValueError('Video generator returned incomplete or diagnostic-only signals; these cannot be used as IMU training measurements')
        samples = np.asarray(payload['samples'], dtype=np.float32)
        if samples.shape != (request['count'], length, channels) or not np.isfinite(samples).all():
            raise ValueError('MCP response violates sample count/shape/finiteness contract')
        return samples

    async def _call(self, spec: dict, args: dict) -> dict:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError('Install the mcp extra for synthetic generation') from exc
        server = StdioServerParameters(command=spec['command'], args=spec.get('args', []),
                                       env=spec.get('env'))
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(spec['tool'], arguments=args)
                if getattr(result, 'is_error', getattr(result, 'isError', False)):
                    raise RuntimeError('MCP generator returned an error')
                structured = getattr(result, 'structured_content', getattr(result, 'structuredContent', None))
                if structured is not None:
                    return structured
                texts = [c.text for c in result.content if c.type == 'text']
                return json.loads('\n'.join(texts))
