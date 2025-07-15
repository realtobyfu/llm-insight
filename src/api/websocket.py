"""WebSocket endpoints for real-time analysis updates"""

import asyncio
import json
from typing import Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..core import InterpretabilityAnalyzer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WSMessage(BaseModel):
    """WebSocket message format"""
    type: str
    data: Dict


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_data: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_data[websocket] = {}
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        self.connection_data.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, websocket: WebSocket, message: WSMessage):
        """Send message to specific client"""
        try:
            await websocket.send_json(message.dict())
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: WSMessage):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message.dict())
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


class RealtimeAnalyzer:
    """Handles real-time analysis with progress updates"""
    
    def __init__(self, analyzer: InterpretabilityAnalyzer, manager: ConnectionManager):
        self.analyzer = analyzer
        self.manager = manager
    
    async def analyze_with_updates(
        self,
        websocket: WebSocket,
        text: str,
        methods: List[str],
        options: Optional[Dict] = None
    ):
        """Perform analysis with real-time progress updates"""
        try:
            # Send initial status
            await self.manager.send_message(
                websocket,
                WSMessage(
                    type="status",
                    data={"message": "Starting analysis...", "progress": 0}
                )
            )
            
            total_methods = len(methods)
            completed = 0
            results = {}
            
            # Process each method with updates
            for method in methods:
                # Update progress
                await self.manager.send_message(
                    websocket,
                    WSMessage(
                        type="progress",
                        data={
                            "message": f"Running {method} analysis...",
                            "progress": int((completed / total_methods) * 100),
                            "current_method": method
                        }
                    )
                )
                
                # Run analysis for single method
                method_result = await asyncio.to_thread(
                    self.analyzer.analyze,
                    text,
                    methods=[method],
                    **(options or {})
                )
                
                # Extract method result
                if method in method_result:
                    results[method] = method_result[method]
                
                # Send intermediate result
                await self.manager.send_message(
                    websocket,
                    WSMessage(
                        type="intermediate_result",
                        data={
                            "method": method,
                            "result": self._serialize_result(results[method])
                        }
                    )
                )
                
                completed += 1
            
            # Add metadata
            results["metadata"] = method_result.get("metadata", {})
            
            # Send final results
            await self.manager.send_message(
                websocket,
                WSMessage(
                    type="complete",
                    data={
                        "message": "Analysis complete",
                        "progress": 100,
                        "results": self._serialize_result(results)
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
            await self.manager.send_message(
                websocket,
                WSMessage(
                    type="error",
                    data={"message": str(e)}
                )
            )
    
    def _serialize_result(self, result):
        """Convert tensors to JSON-serializable format"""
        import torch
        import numpy as np
        
        def serialize_value(value):
            if isinstance(value, torch.Tensor):
                return value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(v) for v in value]
            else:
                return value
        
        return serialize_value(result)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, analyzer: InterpretabilityAnalyzer):
    """Main WebSocket endpoint handler"""
    await manager.connect(websocket)
    realtime_analyzer = RealtimeAnalyzer(analyzer, manager)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Parse message
            message_type = data.get("type")
            
            if message_type == "analyze":
                # Start analysis
                text = data.get("text", "")
                methods = data.get("methods", ["attention", "importance"])
                options = data.get("options", {})
                
                # Run analysis in background
                asyncio.create_task(
                    realtime_analyzer.analyze_with_updates(
                        websocket, text, methods, options
                    )
                )
            
            elif message_type == "ping":
                # Respond to ping
                await manager.send_message(
                    websocket,
                    WSMessage(type="pong", data={})
                )
            
            elif message_type == "subscribe":
                # Subscribe to specific events
                events = data.get("events", [])
                manager.connection_data[websocket]["subscriptions"] = events
                
                await manager.send_message(
                    websocket,
                    WSMessage(
                        type="subscribed",
                        data={"events": events}
                    )
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)