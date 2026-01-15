"""
Query Router for Hybrid RAG System
Routes queries between pandas aggregation and RAG semantic search.
"""

import re
import pandas as pd
from typing import Dict, Tuple


class QueryRouter:
    """
    Intelligent query router that classifies queries and computes aggregations.

    Solves the critical RAG limitation: aggregation queries like "which fund
    performed best" need ALL funds' data, not just top-k semantic chunks.
    """

    def __init__(self, holdings_df: pd.DataFrame, trades_df: pd.DataFrame):
        """Initialize with loaded DataFrames."""
        self.holdings_df = holdings_df
        self.trades_df = trades_df

    def classify_query_type(self, query: str) -> str:
        """
        Classify query type to determine routing strategy.

        Returns:
            'aggregation': Requires all funds comparison (use pandas)
            'specific': Targets specific fund/security (use RAG)
        """
        query_lower = query.lower()

        # Patterns that require ALL funds data (aggregation)
        aggregation_patterns = [
            r'which fund.*best',
            r'which fund.*performed',
            r'which fund.*better',
            r'compare.*funds',
            r'top.*funds',
            r'rank.*funds',
            r'all funds',
            r'fund performance',
            r'best.*yearly.*p[&/]l',
            r'highest.*p[&/]l',
            r'lowest.*p[&/]l',
            r'funds.*better',
            r'funds.*worse',
            r'total.*all funds',
            r'aggregate.*funds'
        ]

        for pattern in aggregation_patterns:
            if re.search(pattern, query_lower):
                return 'aggregation'

        return 'specific'

    def compute_fund_aggregations(self) -> Dict:
        """
        Compute key aggregations for ALL funds in the dataset.

        This ensures completeness - every fund is included in the stats.
        """
        stats = {}

        # Holdings aggregations
        if self.holdings_df is not None and len(self.holdings_df) > 0:
            # Total P&L by fund (CRITICAL for performance ranking)
            pl_by_fund = self.holdings_df.groupby('PortfolioName')['PL_YTD'].sum()
            pl_by_fund = pl_by_fund.sort_values(ascending=False)
            stats['pl_ytd_by_fund'] = pl_by_fund.to_dict()

            # Holdings count by fund
            holdings_count = self.holdings_df.groupby('PortfolioName').size()
            stats['holdings_count'] = holdings_count.to_dict()

            # Average P&L by fund
            avg_pl = self.holdings_df.groupby('PortfolioName')['PL_YTD'].mean()
            stats['avg_pl_ytd'] = avg_pl.to_dict()

            # P&L metrics by fund (QTD, MTD, DTD)
            pl_metrics = self.holdings_df.groupby('PortfolioName').agg({
                'PL_YTD': 'sum',
                'PL_QTD': 'sum',
                'PL_MTD': 'sum',
                'PL_DTD': 'sum'
            }).to_dict()
            stats['pl_metrics'] = pl_metrics

        # Trades aggregations
        if self.trades_df is not None and len(self.trades_df) > 0:
            # Trade count by fund
            trade_count = self.trades_df.groupby('PortfolioName').size()
            stats['trade_count'] = trade_count.to_dict()

            # Buy vs Sell by fund
            try:
                trade_types = self.trades_df.groupby(['PortfolioName', 'TradeTypeName']).size()
                trade_types = trade_types.unstack(fill_value=0)
                stats['trade_types'] = trade_types.to_dict()
            except:
                # Fallback if groupby fails
                stats['trade_types'] = {}

            # Total cash by fund
            cash_by_fund = self.trades_df.groupby('PortfolioName')['TotalCash'].sum()
            stats['total_cash'] = cash_by_fund.to_dict()

        return stats

    def format_aggregation_context(self, stats: Dict, query: str) -> str:
        """
        Format aggregated statistics as context for LLM.

        Returns a complete summary of ALL funds, ensuring no fund is missed.
        """
        context_parts = []

        # P&L rankings (most important for performance queries)
        if 'pl_ytd_by_fund' in stats and stats['pl_ytd_by_fund']:
            context_parts.append("=== COMPLETE Fund Performance Rankings (Yearly P&L) ===")
            context_parts.append("This includes ALL funds in the dataset:\n")

            pl_sorted = sorted(stats['pl_ytd_by_fund'].items(),
                             key=lambda x: x[1], reverse=True)

            for i, (fund, pl) in enumerate(pl_sorted, 1):
                context_parts.append(f"{i}. {fund}: ${pl:,.2f}")

            context_parts.append(f"\nTotal Funds: {len(pl_sorted)}")

        # Holdings count
        if 'holdings_count' in stats and stats['holdings_count']:
            context_parts.append("\n\n=== Holdings Count by Fund ===")
            for fund, count in sorted(stats['holdings_count'].items(),
                                     key=lambda x: x[1], reverse=True):
                context_parts.append(f"  {fund}: {count} holdings")

        # Trade count
        if 'trade_count' in stats and stats['trade_count']:
            context_parts.append("\n\n=== Trade Count by Fund ===")
            for fund, count in sorted(stats['trade_count'].items(),
                                     key=lambda x: x[1], reverse=True):
                context_parts.append(f"  {fund}: {count} trades")

        # Total cash (if query mentions cash/value)
        if 'total' in query.lower() and 'cash' in query.lower():
            if 'total_cash' in stats and stats['total_cash']:
                context_parts.append("\n\n=== Total Cash by Fund ===")
                for fund, cash in sorted(stats['total_cash'].items(),
                                       key=lambda x: x[1], reverse=True):
                    context_parts.append(f"  {fund}: ${cash:,.2f}")

        if not context_parts:
            return "No aggregation data available."

        return "\n".join(context_parts)

    def get_fund_summary(self, fund_name: str) -> str:
        """Get a summary for a specific fund (useful for specific queries)."""
        summary_parts = []

        if self.holdings_df is not None:
            fund_holdings = self.holdings_df[
                self.holdings_df['PortfolioName'] == fund_name
            ]

            if len(fund_holdings) > 0:
                summary_parts.append(f"Fund: {fund_name}")
                summary_parts.append(f"Holdings Count: {len(fund_holdings)}")
                summary_parts.append(f"Total P&L YTD: ${fund_holdings['PL_YTD'].sum():,.2f}")
                summary_parts.append(f"Avg P&L YTD: ${fund_holdings['PL_YTD'].mean():,.2f}")

        if self.trades_df is not None:
            fund_trades = self.trades_df[
                self.trades_df['PortfolioName'] == fund_name
            ]

            if len(fund_trades) > 0:
                summary_parts.append(f"Trade Count: {len(fund_trades)}")
                buy_trades = len(fund_trades[fund_trades['TradeTypeName'] == 'Buy'])
                sell_trades = len(fund_trades[fund_trades['TradeTypeName'] == 'Sell'])
                summary_parts.append(f"  Buy: {buy_trades}, Sell: {sell_trades}")

        return "\n".join(summary_parts) if summary_parts else f"No data found for {fund_name}"


if __name__ == "__main__":
    # Test the query router
    print("🧪 Testing Query Router...\n")

    # Load sample data for testing
    try:
        import sys
        sys.path.append('.')
        from data_processor import DataProcessor

        processor = DataProcessor('holdings.csv', 'trades.csv')
        processor.load_data()
        processor.clean_data()

        # Initialize router
        router = QueryRouter(processor.holdings_df, processor.trades_df)

        # Test query classification
        test_queries = [
            ("Which fund performed best based on yearly P&L?", "aggregation"),
            ("Compare all funds", "aggregation"),
            ("How many holdings does MNC Investment Fund have?", "specific"),
            ("What securities does Garfield hold?", "specific"),
            ("Top funds by performance", "aggregation"),
            ("Total trades for HoldCo 1", "specific")
        ]

        print("=" * 60)
        print("Query Classification Tests")
        print("=" * 60)

        for query, expected in test_queries:
            classified = router.classify_query_type(query)
            status = "✅" if classified == expected else "❌"
            print(f"{status} '{query[:45]}...'")
            print(f"   Expected: {expected}, Got: {classified}\n")

        # Test aggregation computation
        print("\n" + "=" * 60)
        print("Aggregation Computation Test")
        print("=" * 60)

        stats = router.compute_fund_aggregations()
        print(f"\n📊 Computed stats for {len(stats.get('pl_ytd_by_fund', {}))} funds")

        # Format context
        test_query = "Which fund performed best?"
        context = router.format_aggregation_context(stats, test_query)
        print(f"\n📄 Formatted context ({len(context)} chars):")
        print(context[:500] + "..." if len(context) > 500 else context)

        print("\n✅ Query router test complete!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
