import { Camera, Radio, Maximize2, MonitorPlay } from "lucide-react";

function VideoFeed() {

  const date = new Date().toLocaleDateString();
  const time = new Date().toLocaleTimeString();

  return (
    <div className="bg-slate-800 shadow-lg border border-slate-700 mt-6 overflow-hidden">

      {/* Header */}

      <div className="flex justify-between items-center px-6 py-4 border-b border-slate-700">

        <div className="flex items-center gap-3">

          <Camera className="text-cyan-400" size={24} />

          <div>

            <h2 className="text-xl font-semibold">

              Camera 01

            </h2>

            <p className="text-gray-400 text-sm">

              AI Surveillance Zone A

            </p>

          </div>

        </div>

        <div className="flex items-center gap-6">

          <div className="flex items-center gap-2">

            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>

            <span className="text-red-400 font-semibold">

              REC

            </span>

          </div>

          <button className="hover:text-cyan-400">

            <Maximize2 size={22} />

          </button>

        </div>

      </div>

      {/* Video */}

      <div className="relative bg-black h-[500px] flex items-center justify-center">

        {/* Grid Overlay */}

        <div className="absolute inset-0 opacity-10">

          <div className="grid grid-cols-3 h-full">

            <div className="border-r border-white"></div>

            <div className="border-r border-white"></div>

            <div></div>

          </div>

          <div className="absolute inset-0 grid grid-rows-3">

            <div className="border-b border-white"></div>

            <div className="border-b border-white"></div>

            <div></div>

          </div>

        </div>

        {/* Placeholder */}

        <div className="text-center">

          <MonitorPlay
            size={90}
            className="mx-auto text-slate-500"
          />

          <h2 className="text-2xl font-semibold mt-6">

            Live Feed Placeholder

          </h2>

          <p className="text-gray-500 mt-2">

            YOLO Detection Feed will appear here

          </p>

        </div>

        {/* Top Left */}

        <div className="absolute top-5 left-5">

          <div className="bg-black/60 px-3 py-2 rounded-lg">

            <p className="text-sm">

              Camera ID : CAM-01

            </p>

          </div>

        </div>

        {/* Top Right */}

        <div className="absolute top-5 right-5">

          <div className="bg-black/60 px-3 py-2 rounded-lg">

            <p className="text-sm">

              Resolution : 1920 × 1080

            </p>

          </div>

        </div>

        {/* Bottom Left */}

        <div className="absolute bottom-5 left-5">

          <div className="bg-black/60 px-3 py-2 rounded-lg">

            <p className="text-sm">

              30 FPS | GPU Active

            </p>

          </div>

        </div>

        {/* Bottom Right */}

        <div className="absolute bottom-5 right-5">

          <div className="bg-black/60 px-3 py-2 rounded-lg">

            <p className="text-sm">

              {date}

            </p>

            <p className="text-sm">

              {time}

            </p>

          </div>

        </div>

      </div>

    </div>
  );

}

export default VideoFeed;