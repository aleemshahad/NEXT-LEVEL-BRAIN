import tkinter as tk
from tkinter import ttk, messagebox
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
import json
from pathlib import Path

class LivePortfolioDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 NEXT LEVEL BRAIN - LIVE PERFORMANCE & CONTROL")
        self.root.geometry("1200x850")
        self.root.configure(bg='#0a0a0a') # Deeper Dark Background
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._setup_styles()
        
        self.running = True
        self.magic_buy = 777001
        self.magic_sell = 777002
        self.history_days = 30 # Track last 30 days by default
        
        self.trade_history = []
        self.metrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0
        }
        
        self.session_stats_file = Path("logs/dashboard_session_stats.json")
        self.accumulated_seconds = self._load_session_stats()
        self.start_time = time.time()
        self._create_widgets()
        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Start high-frequency timer tick (Every 1 second)
        self._tick_timer()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _load_session_stats(self):
        """Load accumulated trading time from disk"""
        try:
            if self.session_stats_file.exists():
                with open(self.session_stats_file, 'r') as f:
                    return json.load(f).get('accumulated_seconds', 0)
        except Exception: pass
        return 0

    def _save_session_stats(self):
        """Save total trading time to disk"""
        try:
            self.session_stats_file.parent.mkdir(parents=True, exist_ok=True)
            current_session = time.time() - self.start_time
            total = self.accumulated_seconds + current_session
            with open(self.session_stats_file, 'w') as f:
                json.dump({'accumulated_seconds': total}, f)
        except Exception: pass
    def _setup_styles(self):
        self.style.configure("TFrame", background="#0a0a0a")
        self.style.configure("Card.TFrame", background="#151515", relief="flat", borderwidth=0)
        self.style.configure("TLabel", background="#151515", foreground="#ffffff", font=('Segoe UI', 10))
        self.style.configure("Header.TLabel", background="#0a0a0a", foreground="#00e676", font=('Segoe UI', 16, 'bold'))
        self.style.configure("Stat.TLabel", background="#151515", foreground="#00ff00", font=('Consolas', 14, 'bold'))
        self.style.configure("Metric.TLabel", background="#151515", foreground="#ffffff", font=('Segoe UI', 9))
        
        # Treeview styles
        self.style.configure("Treeview", 
                           background="#151515", 
                           foreground="white", 
                           fieldbackground="#151515",
                           rowheight=28,
                           font=('Segoe UI', 10))
        self.style.map("Treeview", background=[('selected', '#3d3d3d')])
        self.style.configure("Treeview.Heading", background="#212121", foreground="white", font=('Segoe UI', 10, 'bold'))

    def _create_widgets(self):
        # Main Container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Header
        header_frame = ttk.Frame(main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title = tk.Label(header_frame, text="🧠 NEXT LEVEL BRAIN - LIVE PERFORMANCE", 
                        bg="#0a0a0a", fg="#00e676", font=('Segoe UI', 22, 'bold'))
        title.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header_frame, text="● SYSTEM ACTIVE", bg="#0a0a0a", 
                                   fg="#00e676", font=('Segoe UI', 10, 'bold'))
        self.status_label.pack(side=tk.RIGHT)

        # 2. Performance Summary Metrics (User Requested View)
        perf_container = ttk.Frame(main_frame, style="Card.TFrame", padding="15")
        perf_container.pack(fill=tk.X, pady=10)
        
        metrics_list = [
            ("Total Trades", "total_trades"),
            ("Win Rate", "win_rate"),
            ("Total P&L", "total_pnl"),
            ("Profit Factor", "profit_factor"),
            ("Max Drawdown", "max_drawdown")
        ]
        
        self.metric_labels = {}
        for i, (label, key) in enumerate(metrics_list):
            m_frame = ttk.Frame(perf_container, style="Card.TFrame")
            m_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            ttk.Label(m_frame, text=f"{label}:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=5)
            self.metric_labels[key] = ttk.Label(m_frame, text="--", style="Stat.TLabel")
            self.metric_labels[key].pack(side=tk.LEFT, padx=5)

        # 3. Portfolio Summary Cards (Real-time)
        summary_container = ttk.Frame(main_frame, style="TFrame")
        summary_container.pack(fill=tk.X, pady=10)
        
        self.cards = {}
        items = [
            ("ACCOUNT BALANCE", "balance_val", "#ffffff"),
            ("FLOATING EQUITY", "equity_val", "#00e676"),
            ("SESSION PNL", "session_val", "#00e676"),
            ("MARGIN LEVEL", "margin_val", "#03a9f4"),
            ("SESSION TIME", "duration_val", "#ffeb3b"),
            ("EQUITY MILESTONE", "milestone_val", "#e91e63")
        ]
        
        for i, (label, key, color) in enumerate(items):
            card = ttk.Frame(summary_container, style="Card.TFrame", padding="15")
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            ttk.Label(card, text=label, font=('Segoe UI', 9, 'bold'), foreground="#888888").pack(anchor=tk.W)
            self.cards[key] = tk.Label(card, text="$0.00", bg="#151515", 
                                      fg=color, font=('Consolas', 20, 'bold'))
            self.cards[key].pack(anchor=tk.W, pady=5)

        # 4. Tabbed View for Positions and History
        tab_control = ttk.Notebook(main_frame)
        
        # Tab 1: Active Positions
        self.tab_pos = ttk.Frame(tab_control, padding="10")
        self.pos_tree = ttk.Treeview(self.tab_pos, columns=("ticket", "symbol", "side", "lots", "price", "profit"), show="headings")
        self.pos_tree.heading("ticket", text="Ticket")
        self.pos_tree.heading("symbol", text="Symbol")
        self.pos_tree.heading("side", text="Side")
        self.pos_tree.heading("lots", text="Lots")
        self.pos_tree.heading("price", text="Entry")
        self.pos_tree.heading("profit", text="Profit ($)")
        for col in ("ticket", "symbol", "side", "lots", "price", "profit"):
            self.pos_tree.column(col, anchor=tk.CENTER, width=100)
        self.pos_tree.pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Recent History
        self.tab_hist = ttk.Frame(tab_control, padding="10")
        self.hist_tree = ttk.Treeview(self.tab_hist, columns=("time", "symbol", "side", "lots", "profit", "comment"), show="headings")
        self.hist_tree.heading("time", text="Time")
        self.hist_tree.heading("symbol", text="Symbol")
        self.hist_tree.heading("side", text="Side")
        self.hist_tree.heading("lots", text="Lots")
        self.hist_tree.heading("profit", text="Profit")
        self.hist_tree.heading("comment", text="Comment")
        for col in ("time", "symbol", "side", "lots", "profit", "comment"):
            self.hist_tree.column(col, anchor=tk.CENTER, width=100)
        self.hist_tree.pack(fill=tk.BOTH, expand=True)

        tab_control.add(self.tab_pos, text=' 💹 ACTIVE POSITIONS ')
        tab_control.add(self.tab_hist, text=' 📜 ALL TRADES HISTORY ')
        tab_control.pack(fill=tk.BOTH, expand=True, pady=10)

        # Footer Actions
        footer = ttk.Frame(main_frame, style="TFrame")
        footer.pack(fill=tk.X, pady=5)
        
        tk.Button(footer, text="🔄 RESET PERFORMANCE", command=self._reset_dashboard, bg="#2196f3", fg="white", font=('Segoe UI', 9, 'bold')).pack(side=tk.RIGHT, padx=5)
        tk.Button(footer, text="🧹 DELETE ALL PENDINGS", command=self._delete_pendings, bg="#ff5252", fg="white", font=('Segoe UI', 9, 'bold')).pack(side=tk.RIGHT, padx=5)
        tk.Button(footer, text="📊 GENERATE FULL REPORT", command=self._generate_report, bg="#00e676", fg="black", font=('Segoe UI', 9, 'bold')).pack(side=tk.RIGHT, padx=5)

    def _delete_pendings(self):
        if not messagebox.askyesno("Confirm", "Close ALL active positions and delete ALL pending orders?"):
            return

        closed_pos = 0
        cancelled_orders = 0

        # Close ALL active positions — each independently so one failure doesn't stop the rest
        try:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    try:
                        sym_info = mt5.symbol_info(pos.symbol)
                        if not sym_info: continue
                        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                        tick = mt5.symbol_info_tick(pos.symbol)
                        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                        done = False
                        for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
                            req = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": pos.symbol,
                                "volume": pos.volume,
                                "type": order_type,
                                "position": pos.ticket,
                                "price": price,
                                "deviation": 50,
                                "magic": pos.magic,
                                "comment": "DASHBOARD_CLOSE_ALL",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": filling,
                            }
                            res = mt5.order_send(req)
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                closed_pos += 1
                                done = True
                                break
                        if not done:
                            print(f"Failed to close pos {pos.ticket} on all filling modes")
                    except Exception as e:
                        print(f"Error closing position {pos.ticket}: {e}")
        except Exception as e:
            print(f"Error getting positions: {e}")

        # Cancel ALL pending orders — each independently so one failure doesn't stop the rest
        try:
            orders = mt5.orders_get()
            if orders:
                for o in orders:
                    try:
                        res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            cancelled_orders += 1
                        else:
                            print(f"Failed to cancel order {o.ticket}: {res.retcode if res else 'None'}")
                    except Exception as e:
                        print(f"Error cancelling order {o.ticket}: {e}")
        except Exception as e:
            print(f"Error getting orders: {e}")

        messagebox.showinfo("Done", f"Closed: {closed_pos} positions\nCancelled: {cancelled_orders} pending orders.")

    def _reset_dashboard(self):
        msg = ("⚠️ RESET PERFORMANCE ⚠️\n\n"
               "This will:\n"
               "1. Reset all performance stats & time\n"
               "2. Reset MILESTONE baseline\n\n"
               "No positions or orders will be touched.\n\n"
               "Are you sure you want to proceed?")

        if not messagebox.askyesno("Confirm Performance Reset", msg):
            return

        # Signal the main script to reset its internal milestone state
        try:
            signal_file = Path("logs/global_reset.signal")
            signal_file.parent.mkdir(parents=True, exist_ok=True)
            with open(signal_file, 'w') as f:
                f.write(str(datetime.now().timestamp()))
            
            # 🧹 CLEAN SWEEP: Delete state files that cause glitches on reset
            files_to_del = [
                "logs/grid_state.json",
                "logs/recycler_state.json",
                "logs/smart_trailing_state.json",
                "logs/milestone_progress.json"
            ]
            
            # Delete profit state files (all that match the pattern)
            for ps in Path("logs").glob("profit_state_*.json"):
                files_to_del.append(str(ps))
                
            for f in files_to_del:
                fp = Path(f)
                if fp.exists():
                    fp.unlink()
                    print(f"Deleted stale log: {f}")

            # Optional: Clean live reports folder if you want a fresh start there too
            report_dir = Path("logs/live_reports")
            if report_dir.exists():
                for f in report_dir.glob("*"):
                    f.unlink()
            
        except Exception as e:
            print(f"Failed to perform clean reset: {e}")

        reset_point = datetime.now().timestamp()
        reset_file = Path("logs/dashboard_reset.json")
        reset_file.parent.mkdir(parents=True, exist_ok=True)

        with open(reset_file, 'w') as f:
            json.dump({'reset_timestamp': reset_point}, f)

        self.trade_history = []
        # Reset Persistent Timer
        self.accumulated_seconds = 0
        self.start_time = time.time()
        self._save_session_stats()

        messagebox.showinfo("Reset Successful",
                            "✅ Performance Reset Complete!\n\nStats & Timer reset.\nPositions and orders are untouched.")
    def _generate_report(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(f"logs/live_reports/dashboard_report_{timestamp}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'trades': self.trade_history
        }
        with open(report_path, 'w') as f:
            json.dump(report_data, f, default=str, indent=4)
        messagebox.showinfo("Report Saved", f"Full performance report saved to:\n{report_path}")

    def _update_loop(self):
        if not mt5.initialize():
            self.root.after(0, lambda: messagebox.showerror("Error", "MT5 not running!"))
            self.running = False
            return

        while self.running:
            try:
                acc = mt5.account_info()

                # Auto-save duration every 30 seconds
                current_session = time.time() - self.start_time
                if int(current_session) % 30 == 0:
                    self._save_session_stats()

                positions = mt5.positions_get()

                # Market status (read once per loop)
                symbol = "XAUUSDm"
                sym_info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol)
                market_closed = False
                if sym_info and sym_info.trade_mode in [mt5.SYMBOL_TRADE_MODE_DISABLED, mt5.SYMBOL_TRADE_MODE_CLOSEONLY]:
                    market_closed = True
                if not market_closed and tick:
                    tick_time = datetime.fromtimestamp(tick.time)
                    if (datetime.now() - tick_time).total_seconds() > 60:
                        market_closed = True

                # ── Schedule ALL widget updates on the main (Tkinter) thread ──
                def _apply_ui(acc=acc, positions=positions,
                               market_closed=market_closed):
                    try:
                        if acc:
                            self.cards['balance_val'].config(text=f"${acc.balance:,.2f}")
                            self.cards['equity_val'].config(text=f"${acc.equity:,.2f}")
                            margin_pct = f"{acc.margin_level:.1f}%" if acc.margin_level else "0%"
                            self.cards['margin_val'].config(text=margin_pct)

                        self._update_positions_tree(positions)
                        self._update_full_history()
                    except Exception as ui_err:
                        print(f"UI apply error: {ui_err}")

                self.root.after(0, _apply_ui)

                time.sleep(1)
            except Exception as e:
                print(f"UI Update error: {e}")
                time.sleep(5)

    def _tick_timer(self):
        """Dedicated high-frequency update for the session timer (1s refresh)"""
        if not self.running: return
        try:
            total_seconds = int(self.accumulated_seconds + (time.time() - self.start_time))
            
            d, r = divmod(total_seconds, 86400)
            h, r = divmod(r, 3600)
            m, s = divmod(r, 60)
            
            if d > 0:
                uptime_str = f"{d}d {h:02d}:{m:02d}:{s:02d}"
            else:
                uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
                
            self.cards['duration_val'].config(text=uptime_str)

            # High-Frequency Equity Milestone Update
            try:
                milestone_file = Path("logs/milestone_progress.json")
                if milestone_file.exists():
                    with open(milestone_file, 'r') as f:
                        data = json.load(f)
                        prog = data.get('progress', 0.0)
                        target = data.get('target_inc', 100.0)
                        color = "#e91e63" if prog >= 0 else "#ff5252"
                        text = f"${prog:,.2f} / ${target:.0f}"
                        self.cards['milestone_val'].config(text=text, fg=color)
            except Exception: pass
        except Exception: pass
        self.root.after(1000, self._tick_timer)
    def _update_positions_tree(self, positions):
        # Selected items tracking if needed
        for i in self.pos_tree.get_children():
            self.pos_tree.delete(i)
            
        if not positions:
            return
            
        for p in positions:
            side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            self.pos_tree.insert("", tk.END, values=(
                p.ticket, p.symbol, side, p.volume, p.price_open, f"{p.profit:.2f}"
            ))

    def _update_full_history(self):
        from_date = datetime.now() - timedelta(days=self.history_days)
        to_date = datetime.now() + timedelta(days=1)
        
        deals = mt5.history_deals_get(from_date, to_date)
        if deals:
            # Persistent Reset Logic
            reset_file = Path("logs/dashboard_reset.json")
            reset_ts = 0
            if reset_file.exists():
                try:
                    with open(reset_file, 'r') as f:
                        reset_ts = json.load(f).get('reset_timestamp', 0)
                except: pass

            closed_deals = [d for d in deals if d.entry == 1 and d.time > reset_ts]
            
            new_history = []
            for d in closed_deals:
                new_history.append({
                    'time': datetime.fromtimestamp(d.time).strftime('%Y-%m-%d %H:%M'),
                    'symbol': d.symbol,
                    'side': 'BUY' if d.type == mt5.DEAL_TYPE_BUY else 'SELL',
                    'volume': d.volume,
                    'profit': d.profit + d.commission + d.swap,
                    'comment': d.comment or ""
                })
            
            if len(new_history) != len(self.trade_history):
                self.trade_history = new_history
                for i in self.hist_tree.get_children(): self.hist_tree.delete(i)
                for item in sorted(self.trade_history, key=lambda x: x['time'], reverse=True)[:50]:
                    self.hist_tree.insert("", tk.END, values=(
                        item['time'], item['symbol'], item['side'], item['volume'], f"{item['profit']:.2f}", item['comment']
                    ))

            profits = [h['profit'] for h in self.trade_history]
            if profits:
                wins = [p for p in profits if p > 0]
                losses = [p for p in profits if p <= 0]
                
                total_pnl = sum(profits)
                win_rate = len(wins) / len(profits) if profits else 0
                
                gross_profit = sum(wins)
                gross_loss = abs(sum(losses))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                cum_pnl = np.cumsum(profits)
                acc_info = mt5.account_info()
                base_balance = acc_info.balance if acc_info else 10000.0
                equity_curve = base_balance + cum_pnl
                peak = np.maximum.accumulate(equity_curve)
                dd_pct = (peak - equity_curve) / peak * 100
                max_dd_pct = np.max(dd_pct) if len(dd_pct) > 0 else 0
                
                self.metrics = {
                    'total_trades': len(profits), 'win_rate': win_rate, 'total_pnl': total_pnl,
                    'profit_factor': profit_factor, 'max_drawdown': max_dd_pct
                }
                
                self.metric_labels['total_trades'].config(text=str(len(profits)))
                self.metric_labels['win_rate'].config(text=f"{win_rate:.1%}")
                pnl_color = "#00ff00" if total_pnl >= 0 else "#ff5252"
                self.metric_labels['total_pnl'].config(text=f"${total_pnl:,.2f}", foreground=pnl_color)
                self.metric_labels['profit_factor'].config(text=f"{profit_factor:.2f}")
                self.metric_labels['max_drawdown'].config(text=f"{max_dd_pct:.2f}%")
                
                # Session PNL (Last 24h)
                now = datetime.now()
                session_pnl = sum(h['profit'] for h in self.trade_history if (now - datetime.strptime(h['time'], '%Y-%m-%d %H:%M')).total_seconds() < 86400)
                session_color = "#00e676" if session_pnl >= 0 else "#ff5252"
                self.cards['session_val'].config(text=f"${session_pnl:,.2f}", fg=session_color)
            else:
                # Reset labels to zero/empty
                self.metric_labels['total_trades'].config(text="0")
                self.metric_labels['win_rate'].config(text="0.0%")
                self.metric_labels['total_pnl'].config(text="$0.00", foreground="#ffffff")
                self.metric_labels['profit_factor'].config(text="0.00")
                self.metric_labels['max_drawdown'].config(text="0.00%")
                self.cards['session_val'].config(text="$0.00", fg="#ffffff")

    def _on_closing(self):
        self.running = False
        self._save_session_stats() # Final save on close
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = LivePortfolioDashboard()
    app.run()
