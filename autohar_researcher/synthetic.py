"""MCP-backed generation. No simulated text/video-to-IMU fallback."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class MCPGenerator:
    # Configured by the experiment owner; agents cannot edit server/tool identity.
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
        payload = asyncio.run(asyncio.wait_for(self._call(spec, args), timeout=spec.get('timeout_seconds', 600)))
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
