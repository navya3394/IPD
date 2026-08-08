import {
  Camera,
  Users,
  ShieldAlert,
  Monitor,
  Cpu,
  ChevronRight,
} from "lucide-react";

function CameraCard() {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-2xl overflow-hidden hover:border-blue-500 transition-all duration-300 shadow-lg">

      {/* Header */}

      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-700">

        <div className="flex items-center gap-3">

          <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">

            <Camera size={20} className="text-blue-400" />

          </div>

          <div>

            <h2 className="text-white font-semibold text-lg">
              Camera 01
            </h2>

            <p className="text-sm text-slate-400">
              Main Entrance
            </p>

          </div>

        </div>

        <span className="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-xs font-semibold">
          ● Online
        </span>

      </div>

      {/* Live Feed Placeholder */}

      <div className="h-52 bg-slate-900 border-y border-slate-700 flex flex-col items-center justify-center">

        <Camera
          size={70}
          className="text-slate-600 mb-4"
        />

        <h3 className="text-xl font-semibold text-white">

          Live Feed

        </h3>

        <p className="text-slate-500 mt-2">

          YOLO feed will appear here

        </p>

      </div>

      {/* Information */}

      <div className="p-5">

        <div className="grid grid-cols-2 gap-y-5">

          <div className="flex items-center gap-3">

            <Users size={18} className="text-blue-400" />

            <div>

              <p className="text-xs text-slate-400">

                People

              </p>

              <h3 className="text-white font-semibold">

                18

              </h3>

            </div>

          </div>

          <div className="flex items-center gap-3">

            <ShieldAlert size={18} className="text-red-400" />

            <div>

              <p className="text-xs text-slate-400">

                Threat Score

              </p>

              <h3 className="text-red-400 font-semibold">

                14%

              </h3>

            </div>

          </div>

          <div className="flex items-center gap-3">

            <Monitor size={18} className="text-cyan-400" />

            <div>

              <p className="text-xs text-slate-400">

                Resolution

              </p>

              <h3 className="text-white font-semibold">

                1920×1080

              </h3>

            </div>

          </div>

          <div className="flex items-center gap-3">

            <Cpu size={18} className="text-green-400" />

            <div>

              <p className="text-xs text-slate-400">

                FPS

              </p>

              <h3 className="text-white font-semibold">

                30

              </h3>

            </div>

          </div>

        </div>

      </div>

      {/* Footer */}

      <div className="border-t border-slate-700 px-5 py-4">

        <button className="w-full bg-blue-600 hover:bg-blue-700 transition rounded-xl py-3 flex justify-center items-center gap-2 font-medium">

          View Details

          <ChevronRight size={18} />

        </button>

      </div>

    </div>
  );
}

export default CameraCard;