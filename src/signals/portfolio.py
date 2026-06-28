import pandas as pd
from datetime import datetime

class PortfolioTracker:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = [{"timestamp": datetime.now(), "equity": initial_capital}]

    def execute_signal(self, signal, price, shares_pct=0.25):
        ticker = signal["ticker"]
        action = signal["signal"]

        if action in ("Strong Buy", "Buy") and ticker not in self.positions:
            cost = min(self.cash * shares_pct * signal["strength"], self.cash)
            shares = cost / price
            self.cash -= cost
            self.positions[ticker] = {"shares": shares, "entry_price": price}
            self.trade_history.append({"timestamp": datetime.now(), "ticker": ticker,
                "action": "BUY", "shares": round(shares, 4), "price": price,
                "value": round(cost, 2), "signal": action})

        elif action in ("Strong Sell", "Sell") and ticker in self.positions:
            pos = self.positions.pop(ticker)
            proceeds = pos["shares"] * price
            self.cash += proceeds
            self.trade_history.append({"timestamp": datetime.now(), "ticker": ticker,
                "action": "SELL", "shares": round(pos["shares"], 4), "price": price,
                "value": round(proceeds, 2), "pnl": round(proceeds - pos["shares"] * pos["entry_price"], 2),
                "pnl_pct": round((price / pos["entry_price"] - 1) * 100, 2), "signal": action})

    def get_portfolio_value(self, current_prices):
        return self.cash + sum(pos["shares"] * current_prices.get(t, pos["entry_price"])
                               for t, pos in self.positions.items())

    def update_equity(self, current_prices):
        self.equity_curve.append({"timestamp": datetime.now(),
                                   "equity": self.get_portfolio_value(current_prices)})

    def get_performance_summary(self, current_prices):
        current_value = self.get_portfolio_value(current_prices)
        trades_df = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        wins = len(trades_df[(trades_df["action"] == "SELL") & (trades_df.get("pnl", pd.Series([0]*len(trades_df))) > 0)]) if not trades_df.empty else 0
        losses = len(trades_df[(trades_df["action"] == "SELL") & (trades_df.get("pnl", pd.Series([0]*len(trades_df))) < 0)]) if not trades_df.empty else 0
        return {
            "initial_capital": self.initial_capital,
            "current_value": round(current_value, 2),
            "cash": round(self.cash, 2),
            "total_return": round(current_value - self.initial_capital, 2),
            "total_return_pct": round((current_value - self.initial_capital) / self.initial_capital * 100, 2),
            "open_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "winning_trades": wins, "losing_trades": losses,
            "positions_detail": pd.DataFrame([{"ticker": t, "shares": round(p["shares"], 4),
                "entry_price": p["entry_price"],
                "unrealized_pnl": round((current_prices.get(t, p["entry_price"]) - p["entry_price"]) * p["shares"], 2)}
                for t, p in self.positions.items()]) if self.positions else pd.DataFrame(),
            "equity_curve": pd.DataFrame(self.equity_curve),
            "trades": trades_df.sort_values("timestamp", ascending=False) if not trades_df.empty else pd.DataFrame(),
        }

    def reset(self, capital=100000.0):
        self.__init__(capital)