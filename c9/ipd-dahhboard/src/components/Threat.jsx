import { ShieldCheck, User, Clock3 } from "lucide-react";

function ThreatStatus() {
  return (
    <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6 shadow-lg">

      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold">Threat Detection</h2>
          <p className="text-gray-400 text-sm">
            Current AI Detection Status
          </p>
        </div>

        <ShieldCheck size={30} className="text-green-400" />
      </div>

      {/* Status */}
      <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4 flex items-center justify-between">

        <div>
          <p className="text-gray-400 text-sm">Current Status</p>

          <h1 className="text-3xl font-bold text-green-400 mt-1">
            SAFE
          </h1>
        </div>

        <div className="w-4 h-4 bg-green-400 rounded-full animate-pulse"></div>

      </div>

      {/* Stats */}

      <div className="grid grid-cols-2 gap-4 mt-6">

        <div className="bg-slate-700 rounded-xl p-4">

          <p className="text-gray-400 text-sm">
            Confidence
          </p>

          <h2 className="text-2xl font-bold mt-2">
            98%
          </h2>

        </div>

        <div className="bg-slate-700 rounded-xl p-4">

          <p className="text-gray-400 text-sm">
            People
          </p>

          <div className="flex items-center gap-2 mt-2">

            <User size={18} />

            <span className="text-2xl font-bold">
              15
            </span>

          </div>

        </div>

      </div>

      {/* Footer */}

      <div className="flex items-center gap-2 mt-6 text-gray-400 text-sm">

        <Clock3 size={16} />

        Last Scan : 10:58 PM

      </div>

    </div>
  );
}

export default ThreatStatus;