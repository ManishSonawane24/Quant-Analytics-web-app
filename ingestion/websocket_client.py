import asyncio
import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

import websockets


BINANCE_FUTURES_WS = "wss://fstream.binance.com/stream"

# Reference: See reference_binance_collector.html for the original HTML collector
# that demonstrates the Binance WebSocket format and normalization logic.


@dataclass
class Trade:
    """
    Normalized Binance futures trade tick.

    Binance `@trade` payload (trades stream) example:
        {
          "e": "trade",     // Event type
          "E": 123456789,   // Event time
          "s": "BTCUSDT",   // Symbol
          "t": 12345,       // Trade ID
          "p": "0.001",     // Price
          "q": "100",       // Quantity
          "b": 88,          // Buyer order ID
          "a": 50,          // Seller order ID
          "T": 123456785,   // Trade time
          "m": true,        // Is the buyer the market maker?
          "M": true         // Ignore
        }
    
    We normalize this to {timestamp, symbol, price, size}, matching the format
    from reference_binance_collector.html:
      - timestamp: T (trade time in ms) or E (event time) as fallback
      - symbol: s (uppercased)
      - price: p (converted to float)
      - size: q (quantity, converted to float)
    """

    timestamp: int  # epoch ms (Binance T)
    symbol: str
    price: float
    size: float


OnTradeCallback = Callable[[Trade], Awaitable[None]]


class BinanceFuturesClient:
    """
    Minimal Binance Futures WebSocket client for trade ticks.

    This client:
      - Connects to the `symbol@trade` multiplexed stream
      - Normalizes payloads into `Trade`
      - Pushes them to an async callback for ingestion/storage

    It is designed to run forever in a background asyncio task.
    """

    def __init__(
        self,
        symbols: List[str],
        on_trade: OnTradeCallback,
        reconnect_delay: float = 5.0,
    ) -> None:
        if not symbols:
            raise ValueError("At least one symbol must be provided")

        # Binance futures symbols are uppercase, but the stream name is lowercase.
        self.symbols = [s.upper() for s in symbols]
        self.streams = "/".join(f"{s.lower()}@trade" for s in self.symbols)
        self.on_trade = on_trade
        self.reconnect_delay = reconnect_delay
        self._running = False

    async def _handle_message(self, message: str) -> None:
        """
        Handle incoming WebSocket message.
        
        Multiplexed stream format: { "stream": "btcusdt@trade", "data": { ... } }
        Single stream format: { "e": "trade", "s": "BTCUSDT", "T": ..., "p": ..., "q": ... }
        
        Normalization matches reference_binance_collector.html:
          - timestamp: T (trade time ms) or E (event time) as fallback
          - symbol: s (uppercased)
          - price: p (converted to float)
          - size: q (quantity, converted to float)
        """
        data = json.loads(message)
        # Multiple stream format: { "stream": "btcusdt@trade", "data": { ... } }
        payload: Dict = data.get("data", {})
        if not payload or payload.get("e") != "trade":
            return

        trade = Trade(
            timestamp=int(payload.get("T", payload.get("E", 0))),
            symbol=payload.get("s", "").upper(),
            price=float(payload.get("p", "0")),
            size=float(payload.get("q", "0")),
        )
        await self.on_trade(trade)

    async def run_forever(self) -> None:
        """
        Connect to Binance and keep streaming trades.

        Includes simple automatic reconnection with a backoff delay.
        """
        self._running = True
        query = f"{BINANCE_FUTURES_WS}?streams={self.streams}"

        while self._running:
            try:
                async with websockets.connect(query, ping_interval=20, ping_timeout=20) as ws:
                    async for msg in ws:
                        await self._handle_message(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                # In production you'd want structured logging and alerting here.
                await asyncio.sleep(self.reconnect_delay)

    def stop(self) -> None:
        self._running = False



