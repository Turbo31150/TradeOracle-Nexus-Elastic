export default function Home() {
  return (
    <main className="min-h-screen bg-black text-green-400 flex flex-col items-center justify-center p-8">
      <div className="text-center space-y-6">
        <h1 className="text-6xl font-bold tracking-wider animate-pulse">
          <span className="text-cyan-400">Trade</span>
          <span className="text-purple-500">Oracle</span>
          <span className="text-white ml-3">Nexus</span>
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl">
          Autonomous AI Trading Agent powered by Elasticsearch
        </p>
        <div className="flex gap-4 justify-center mt-8">
          <div className="border border-cyan-500/30 rounded-lg p-4 bg-cyan-500/5">
            <h3 className="text-cyan-400 font-semibold">SearchNews</h3>
            <p className="text-sm text-gray-500">RAG-powered financial news</p>
          </div>
          <div className="border border-purple-500/30 rounded-lg p-4 bg-purple-500/5">
            <h3 className="text-purple-400 font-semibold">AnalyzeChart</h3>
            <p className="text-sm text-gray-500">Technical analysis engine</p>
          </div>
          <div className="border border-green-500/30 rounded-lg p-4 bg-green-500/5">
            <h3 className="text-green-400 font-semibold">SendAlert</h3>
            <p className="text-sm text-gray-500">Telegram signal alerts</p>
          </div>
        </div>
        <div className="mt-12 text-sm text-gray-600">
          Built for the Elasticsearch Agent Builder Hackathon
        </div>
      </div>
    </main>
  );
}
