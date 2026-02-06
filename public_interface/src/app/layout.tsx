import type { Metadata } from "next";
import "../styles/globals.css";

export const metadata: Metadata = {
  title: "TradeOracle Nexus",
  description: "Autonomous AI Trading Agent powered by Elasticsearch",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-black antialiased">{children}</body>
    </html>
  );
}
