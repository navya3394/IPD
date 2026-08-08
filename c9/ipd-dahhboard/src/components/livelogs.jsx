import {
  CheckCircle,
  AlertTriangle,
  Camera,
  Bell,
} from "lucide-react";

const events = [
  {
    icon: <CheckCircle size={18} className="text-green-400" />,
    title: "Person Detected",
    time: "10:12 PM",
    color: "border-green-500",
  },
  {
    icon: <Camera size={18} className="text-blue-400" />,
    title: "Camera Connected",
    time: "10:15 PM",
    color: "border-blue-500",
  },
  {
    icon: <AlertTriangle size={18} className="text-yellow-400" />,
    title: "Suspicious Motion",
    time: "10:18 PM",
    color: "border-yellow-500",
  },
  {
    icon: <Bell size={18} className="text-red-400" />,
    title: "Threat Detected",
    time: "10:20 PM",
    color: "border-red-500",
  },
];

function LiveLogs() {
  return (
    <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-lg p-6">

      <div className="flex justify-between items-center mb-6">

        <div>

          <h2 className="text-xl font-bold">
            Event Timeline
          </h2>

          <p className="text-gray-400 text-sm">
            Latest Detection Events
          </p>

        </div>

        <span className="text-sm text-green-400">
          Live
        </span>

      </div>

      <div className="space-y-4 max-h-[310px] overflow-y-auto">

        {events.map((event, index) => (

          <div
            key={index}
            className={`bg-slate-700 rounded-xl border-l-4 ${event.color} p-4`}
          >

            <div className="flex justify-between items-center">

              <div className="flex items-center gap-3">

                {event.icon}

                <div>

                  <h3 className="font-semibold">

                    {event.title}

                  </h3>

                  <p className="text-xs text-gray-400">

                    AI Surveillance System

                  </p>

                </div>

              </div>

              <span className="text-gray-400 text-sm">

                {event.time}

              </span>

            </div>

          </div>

        ))}

      </div>

    </div>
  );
}

export default LiveLogs;