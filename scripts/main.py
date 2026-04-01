import json
import math
from typing import Dict, List, Optional
from json import JSONEncoder

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class ConversionObservation:
    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff


class Observation:
    def __init__(self, plainValueObservations, conversionObservations):
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(self, symbol, price, quantity, buyer=None, seller=None, timestamp=0):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp


class TradingState(object):
    def __init__(self, traderData, timestamp, listings, order_depths, own_trades, market_trades, position, observations):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state, orders, conversions, trader_data):
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            ""
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state, trader_data):
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return compressed

    def compress_observations(self, observations):
        co = {}
        for p, o in observations.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders):
        compressed = []
        for arr in orders.values():
            for o in arr:
                compressed.append([o.symbol, o.price, o.quantity])
        return compressed

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value, max_length):
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


class HybridMarketMaker:
    def __init__(
        self,
        symbol: str,
        position_limit: int,
        base_size: int,
        take_size: int,
        history_len: int,
        short_window: int,
        long_window: int,
        signal_threshold: float,
        alpha_scale: float,
        inv_skew: float,
        quote_skew: float,
        take_edge: float,
        join_improve: int,
    ):
        self.symbol = symbol
        self.position_limit = position_limit
        self.base_size = base_size
        self.take_size = take_size
        self.history_len = history_len
        self.short_window = short_window
        self.long_window = long_window
        self.signal_threshold = signal_threshold
        self.alpha_scale = alpha_scale
        self.inv_skew = inv_skew
        self.quote_skew = quote_skew
        self.take_edge = take_edge
        self.join_improve = join_improve

    def best_bid_ask(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None
        return max(order_depth.buy_orders.keys()), min(order_depth.sell_orders.keys())

    def microprice(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None

        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = abs(order_depth.sell_orders[best_ask])

        if bid_vol <= 0 or ask_vol <= 0:
            return (best_bid + best_ask) / 2.0

        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def inventory_ratio(self, position: int) -> float:
        if self.position_limit == 0:
            return 0.0
        return position / self.position_limit

    def avg(self, arr: List[float]) -> float:
        return sum(arr) / len(arr) if arr else 0.0

    def signal(self, mids: List[float]) -> float:
        if len(mids) < self.long_window:
            return 0.0
        short_ma = self.avg(mids[-self.short_window:])
        long_ma = self.avg(mids[-self.long_window:])
        raw = short_ma - long_ma

        if abs(raw) < self.signal_threshold:
            return 0.0
        return raw

    def fair_value(self, raw_mid: float, sig: float, position: int) -> float:
        return raw_mid + self.alpha_scale * sig - self.inv_skew * self.inventory_ratio(position)

    def take_orders(self, orders: List[Order], order_depth: OrderDepth, fair: float, position: int) -> int:
        pos = position

        for ask_px in sorted(order_depth.sell_orders.keys()):
            if ask_px > fair - self.take_edge:
                break
            ask_qty = abs(order_depth.sell_orders[ask_px])
            capacity = self.position_limit - pos
            qty = min(self.take_size, ask_qty, capacity)
            if qty > 0:
                orders.append(Order(self.symbol, ask_px, qty))
                pos += qty
                logger.print(f"TAKE BUY {self.symbol} {qty} @ {ask_px} fair={fair:.2f} pos={pos}")

        for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_px < fair + self.take_edge:
                break
            bid_qty = order_depth.buy_orders[bid_px]
            capacity = self.position_limit + pos
            qty = min(self.take_size, bid_qty, capacity)
            if qty > 0:
                orders.append(Order(self.symbol, bid_px, -qty))
                pos -= qty
                logger.print(f"TAKE SELL {self.symbol} {qty} @ {bid_px} fair={fair:.2f} pos={pos}")

        return pos

    def quote_prices(self, best_bid: int, best_ask: int, fair: float, position: int, sig: float):
        spread = best_ask - best_bid
        inv = self.inventory_ratio(position)

        bias = 0
        if sig > 0:
            bias = 1
        elif sig < 0:
            bias = -1

        bid_anchor = fair - 1 - self.quote_skew * inv
        ask_anchor = fair + 1 - self.quote_skew * inv

        if spread >= 2:
            bid_px = min(best_bid + self.join_improve, math.floor(bid_anchor) + max(0, bias))
            ask_px = max(best_ask - self.join_improve, math.ceil(ask_anchor) + min(0, bias))
        else:
            bid_px = min(best_bid, math.floor(bid_anchor))
            ask_px = max(best_ask, math.ceil(ask_anchor))

        bid_px = int(bid_px)
        ask_px = int(ask_px)

        if bid_px >= ask_px:
            bid_px = best_bid
            ask_px = best_ask

        return bid_px, ask_px

    def quote_sizes(self, position: int, sig: float):
        inv = self.inventory_ratio(position)

        bid_size = self.base_size
        ask_size = self.base_size

        if sig > 0:
            bid_size = int(round(self.base_size * 1.25))
            ask_size = int(round(self.base_size * 0.85))
        elif sig < 0:
            bid_size = int(round(self.base_size * 0.85))
            ask_size = int(round(self.base_size * 1.25))

        if inv > 0.5:
            bid_size = int(round(bid_size * 0.60))
            ask_size = int(round(ask_size * 1.25))
        elif inv < -0.5:
            bid_size = int(round(bid_size * 1.25))
            ask_size = int(round(ask_size * 0.60))

        return max(1, bid_size), max(1, ask_size)

    def run(self, state: TradingState, memory: Dict) -> List[Order]:
        orders: List[Order] = []
        order_depth = state.order_depths.get(self.symbol)
        if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid, best_ask = self.best_bid_ask(order_depth)
        raw_mid = self.microprice(order_depth)
        if raw_mid is None:
            return orders

        symbol_mem = memory.setdefault(self.symbol, {"mids": []})
        mids = symbol_mem["mids"]
        mids.append(raw_mid)
        if len(mids) > self.history_len:
            mids.pop(0)

        position = state.position.get(self.symbol, 0)
        sig = self.signal(mids)
        fair = self.fair_value(raw_mid, sig, position)

        position = self.take_orders(orders, order_depth, fair, position)
        fair = self.fair_value(raw_mid, sig, position)

        bid_px, ask_px = self.quote_prices(best_bid, best_ask, fair, position, sig)
        bid_size, ask_size = self.quote_sizes(position, sig)

        can_buy = self.position_limit - position
        can_sell = self.position_limit + position

        bid_qty = min(bid_size, can_buy)
        ask_qty = min(ask_size, can_sell)

        if bid_qty > 0:
            orders.append(Order(self.symbol, bid_px, bid_qty))
            logger.print(f"BID {self.symbol} {bid_qty} @ {bid_px} sig={sig:.3f} fair={fair:.2f} pos={position}")

        if ask_qty > 0:
            orders.append(Order(self.symbol, ask_px, -ask_qty))
            logger.print(f"ASK {self.symbol} {ask_qty} @ {ask_px} sig={sig:.3f} fair={fair:.2f} pos={position}")

        return orders


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    CONFIG = {
        "EMERALDS": {
            "base_size": 18,
            "take_size": 20,
            "history_len": 20,
            "short_window": 3,
            "long_window": 8,
            "signal_threshold": 0.12,
            "alpha_scale": 1.0,
            "inv_skew": 6.5,
            "quote_skew": 3.5,
            "take_edge": 0.0,
            "join_improve": 1,
        },
        "TOMATOES": {
            "base_size": 36,
            "take_size": 24,
            "history_len": 20,
            "short_window": 3,
            "long_window": 8,
            "signal_threshold": 0.08,
            "alpha_scale": 1.8,
            "inv_skew": 6.5,
            "quote_skew": 4.0,
            "take_edge": 0.0,
            "join_improve": 1,
        },
    }

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {s: [] for s in state.order_depths}
        conversions = 0

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        for symbol, cfg in self.CONFIG.items():
            if symbol not in state.order_depths:
                continue

            bot = HybridMarketMaker(
                symbol=symbol,
                position_limit=self.POSITION_LIMITS[symbol],
                base_size=cfg["base_size"],
                take_size=cfg["take_size"],
                history_len=cfg["history_len"],
                short_window=cfg["short_window"],
                long_window=cfg["long_window"],
                signal_threshold=cfg["signal_threshold"],
                alpha_scale=cfg["alpha_scale"],
                inv_skew=cfg["inv_skew"],
                quote_skew=cfg["quote_skew"],
                take_edge=cfg["take_edge"],
                join_improve=cfg["join_improve"],
            )

            orders = bot.run(state, trader_data)
            result[symbol].extend(orders)

        encoded_trader_data = json.dumps(trader_data)
        logger.flush(state, result, conversions, encoded_trader_data)
        return result, conversions, encoded_trader_data