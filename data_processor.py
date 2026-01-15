"""
Data Processor for Financial Chatbot
Handles CSV loading, cleaning, and chunking for embeddings.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import tiktoken


class DataProcessor:
    """Process holdings and trades CSV files for RAG system."""

    def __init__(self, holdings_path: str, trades_path: str):
        """Initialize with paths to CSV files."""
        self.holdings_path = holdings_path
        self.trades_path = trades_path
        self.holdings_df = None
        self.trades_df = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both CSV files with proper parsing."""
        print("📂 Loading CSV files...")

        # Load holdings with date parsing
        self.holdings_df = pd.read_csv(
            self.holdings_path,
            parse_dates=['AsOfDate', 'OpenDate', 'CloseDate'],
            date_format='%d/%m/%y',
            na_values=['NULL', 'null', '']
        )

        # Load trades
        self.trades_df = pd.read_csv(
            self.trades_path,
            na_values=['NULL', 'null', '']
        )

        print(f"✅ Loaded {len(self.holdings_df)} holdings and {len(self.trades_df)} trades")
        return self.holdings_df, self.trades_df

    def clean_data(self):
        """Clean and standardize data."""
        print("🧹 Cleaning data...")

        # Holdings cleaning
        if self.holdings_df is not None:
            # Fill NaN values
            self.holdings_df['CloseDate'] = self.holdings_df['CloseDate'].fillna('Open')
            self.holdings_df.fillna({'DirectionName': 'Unknown', 'SecurityTypeName': 'Unknown'}, inplace=True)

            # Convert numeric columns
            numeric_cols = ['PL_DTD', 'PL_QTD', 'PL_MTD', 'PL_YTD', 'MV_Local', 'MV_Base', 'Qty', 'Price']
            for col in numeric_cols:
                if col in self.holdings_df.columns:
                    self.holdings_df[col] = pd.to_numeric(self.holdings_df[col], errors='coerce').fillna(0)

        # Trades cleaning
        if self.trades_df is not None:
            # Fill NaN values
            self.trades_df.fillna({'TradeDate': '00:00.0', 'SettleDate': '00:00.0'}, inplace=True)
            self.trades_df.fillna({'TradeTypeName': 'Unknown', 'SecurityType': 'Unknown'}, inplace=True)

            # Convert numeric columns
            numeric_cols = ['Quantity', 'Price', 'Principal', 'TotalCash', 'AllocationCash']
            for col in numeric_cols:
                if col in self.trades_df.columns:
                    self.trades_df[col] = pd.to_numeric(self.trades_df[col], errors='coerce').fillna(0)

        print("✅ Data cleaning complete")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def create_holdings_chunk_text(self, holdings_subset: pd.DataFrame) -> str:
        """Convert holdings DataFrame subset to formatted text."""
        chunks = []

        for _, row in holdings_subset.iterrows():
            chunk = f"""Security: {row['SecName']} ({row['SecurityTypeName']})
Portfolio: {row['PortfolioName']}
Quantity: {row['Qty']:,.0f}
Price: ${row['Price']:,.2f}
Market Value (Base): ${row['MV_Base']:,.2f}
P&L Year-to-Date: ${row['PL_YTD']:,.2f}
P&L Quarter-to-Date: ${row['PL_QTD']:,.2f}
Strategy: {row['Strategy1RefShortName']}
Custodian: {row['CustodianName']}
Direction: {row['DirectionName']}
Open Date: {row['OpenDate']}
---"""
            chunks.append(chunk)

        return "\n".join(chunks)

    def create_trades_chunk_text(self, trades_subset: pd.DataFrame) -> str:
        """Convert trades DataFrame subset to formatted text."""
        chunks = []

        for _, row in trades_subset.iterrows():
            chunk = f"""Trade Type: {row['TradeTypeName']}
Security: {row['Name']} ({row['SecurityType']})
Portfolio: {row['PortfolioName']}
Quantity: {row['Quantity']:,.0f}
Price: ${row['Price']:,.2f}
Total Cash: ${row['TotalCash']:,.2f}
Principal: ${row['Principal']:,.2f}
Trade Date: {row['TradeDate']}
Strategy: {row['Strategy1Name']}
Custodian: {row['CustodianName']}
Counterparty: {row['Counterparty']}
---"""
            chunks.append(chunk)

        return "\n".join(chunks)

    def chunk_by_fund(self, df: pd.DataFrame, fund_name: str,
                      file_type: str, max_tokens: int = 500) -> List[Dict]:
        """Create chunks for a specific fund with token limit."""
        fund_data = df[df['PortfolioName'] == fund_name].copy()

        if len(fund_data) == 0:
            return []

        chunks = []
        current_rows = []
        current_tokens = 0

        for idx, row in fund_data.iterrows():
            # Create text for this row
            if file_type == 'holdings':
                row_text = self.create_holdings_chunk_text(pd.DataFrame([row]))
            else:
                row_text = self.create_trades_chunk_text(pd.DataFrame([row]))

            row_tokens = self.count_tokens(row_text)

            # Check if adding this row exceeds token limit
            if current_tokens + row_tokens > max_tokens and len(current_rows) > 0:
                # Save current chunk
                if file_type == 'holdings':
                    chunk_text = self.create_holdings_chunk_text(pd.DataFrame(current_rows))
                else:
                    chunk_text = self.create_trades_chunk_text(pd.DataFrame(current_rows))

                chunks.append({
                    'text': chunk_text,
                    'metadata': self._create_metadata(current_rows, fund_name, file_type)
                })

                # Reset for new chunk
                current_rows = []
                current_tokens = 0

            current_rows.append(row)
            current_tokens += row_tokens

        # Add remaining rows as final chunk
        if current_rows:
            if file_type == 'holdings':
                chunk_text = self.create_holdings_chunk_text(pd.DataFrame(current_rows))
            else:
                chunk_text = self.create_trades_chunk_text(pd.DataFrame(current_rows))

            chunks.append({
                'text': chunk_text,
                'metadata': self._create_metadata(current_rows, fund_name, file_type)
            })

        return chunks

    def _create_metadata(self, rows: List, fund_name: str, file_type: str) -> Dict:
        """Create metadata for a chunk."""
        metadata = {
            'fund': fund_name,
            'file': file_type,
            'row_count': len(rows)
        }

        if file_type == 'holdings' and rows:
            # Extract security types
            security_types = list(set([r['SecurityTypeName'] for r in rows if 'SecurityTypeName' in r]))
            metadata['security_types'] = security_types[:5]  # Limit to 5 for metadata size

            # Check if has P&L data
            has_pl = any(r.get('PL_YTD', 0) != 0 for r in rows)
            metadata['has_pl'] = has_pl

            # Year from AsOfDate
            if 'AsOfDate' in rows[0] and pd.notna(rows[0]['AsOfDate']):
                try:
                    metadata['year'] = pd.to_datetime(rows[0]['AsOfDate']).year
                except:
                    metadata['year'] = 2023
            else:
                metadata['year'] = 2023

        elif file_type == 'trades' and rows:
            # Extract trade types
            trade_types = list(set([r['TradeTypeName'] for r in rows if 'TradeTypeName' in r]))
            metadata['trade_types'] = trade_types[:5]

            # Security types
            security_types = list(set([r['SecurityType'] for r in rows if 'SecurityType' in r]))
            metadata['security_types'] = security_types[:5]

        return metadata

    def process_all_data(self, max_tokens: int = 500) -> Tuple[List[Dict], List[Dict]]:
        """Process all data and create chunks for all funds."""
        print("🔄 Creating chunks for all funds...")

        # Get unique fund names
        holdings_funds = self.holdings_df['PortfolioName'].unique() if self.holdings_df is not None else []
        trades_funds = self.trades_df['PortfolioName'].unique() if self.trades_df is not None else []

        all_funds = set(list(holdings_funds) + list(trades_funds))

        print(f"📊 Found {len(all_funds)} unique funds")

        holdings_chunks = []
        trades_chunks = []

        for fund in all_funds:
            # Process holdings
            if self.holdings_df is not None and fund in holdings_funds:
                fund_holdings_chunks = self.chunk_by_fund(
                    self.holdings_df, fund, 'holdings', max_tokens
                )
                holdings_chunks.extend(fund_holdings_chunks)

            # Process trades
            if self.trades_df is not None and fund in trades_funds:
                fund_trades_chunks = self.chunk_by_fund(
                    self.trades_df, fund, 'trades', max_tokens
                )
                trades_chunks.extend(fund_trades_chunks)

        print(f"✅ Created {len(holdings_chunks)} holdings chunks and {len(trades_chunks)} trades chunks")
        print(f"📦 Total chunks: {len(holdings_chunks) + len(trades_chunks)}")

        return holdings_chunks, trades_chunks

    def get_summary_stats(self) -> Dict:
        """Generate summary statistics for the datasets."""
        stats = {}

        if self.holdings_df is not None:
            stats['holdings'] = {
                'total_rows': len(self.holdings_df),
                'unique_funds': self.holdings_df['PortfolioName'].nunique(),
                'unique_securities': self.holdings_df['SecurityId'].nunique(),
                'total_pl_ytd': self.holdings_df['PL_YTD'].sum(),
                'security_types': self.holdings_df['SecurityTypeName'].value_counts().to_dict()
            }

            # Top funds by P&L
            top_funds = self.holdings_df.groupby('PortfolioName')['PL_YTD'].sum().sort_values(ascending=False).head(5)
            stats['holdings']['top_funds_by_pl'] = top_funds.to_dict()

        if self.trades_df is not None:
            stats['trades'] = {
                'total_rows': len(self.trades_df),
                'unique_funds': self.trades_df['PortfolioName'].nunique(),
                'buy_trades': len(self.trades_df[self.trades_df['TradeTypeName'] == 'Buy']),
                'sell_trades': len(self.trades_df[self.trades_df['TradeTypeName'] == 'Sell']),
                'total_cash': self.trades_df['TotalCash'].sum(),
                'security_types': self.trades_df['SecurityType'].value_counts().to_dict()
            }

        return stats


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor('holdings.csv', 'trades.csv')
    processor.load_data()
    processor.clean_data()

    # Get stats
    stats = processor.get_summary_stats()
    print("\n📈 Summary Statistics:")
    print(f"Holdings: {stats['holdings']['total_rows']} rows, {stats['holdings']['unique_funds']} funds")
    print(f"Trades: {stats['trades']['total_rows']} rows, {stats['trades']['unique_funds']} funds")

    # Create chunks
    holdings_chunks, trades_chunks = processor.process_all_data()
    print(f"\n✅ Successfully created {len(holdings_chunks) + len(trades_chunks)} chunks")
