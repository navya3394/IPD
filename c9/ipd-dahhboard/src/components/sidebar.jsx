import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Camera,
  ShieldAlert,
  FileText,
  BarChart3,
  Settings,
  Activity,
} from "lucide-react";

function Sidebar() {
  return (
    <aside className="w-72 bg-slate-800 border-r border-slate-700 flex flex-col justify-between">

      {/* Logo */}
      <div>

        <div className="p-8 border-b border-slate-700">

          <h2 className="text-2xl font-bold text-white">
            🛡 AI SURVEILLANCE
          </h2>

          <p className="text-sm text-gray-400 mt-2">
            Security Operations Center
          </p>

        </div>

        {/* Navigation */}

        <div className="mt-8 px-5">

          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `w-full flex items-center gap-4 rounded-xl p-4 mb-4 transition-all duration-300 ${
                isActive
                  ? "bg-blue-600"
                  : "hover:bg-slate-700"
              }`
            }
            >
            <LayoutDashboard size={22} />
            Dashboard
          </NavLink>

          <NavLink
            to="/cameras"
            className={({ isActive }) =>
              `w-full flex items-center gap-4 rounded-xl p-4 mb-4 transition-all duration-300 ${
                isActive
                  ? "bg-blue-600"
                  : "hover:bg-slate-700"
              }`
            }
            >
            <Camera size={22} />
            Cameras
          </NavLink>

          <NavLink
            to="/threat-alerts"
            className={({ isActive }) =>
              `w-full flex items-center gap-4 rounded-xl p-4 mb-4 transition-all duration-300 ${
                isActive
                  ? "bg-blue-600"
                  : "hover:bg-slate-700"
              }`
            }
            >
            <ShieldAlert size={22} />
            Threat Alerts
          </NavLink>

          <NavLink
            to="/event-logs"
            className={({ isActive }) =>
              `w-full flex items-center gap-4 rounded-xl p-4 mb-4 transition-all duration-300 ${
                isActive
                  ? "bg-blue-600"
                  : "hover:bg-slate-700"
              }`
            }
            >
            <FileText size={22} />
            Event Logs
          </NavLink>

          <NavLink
            to="/analytics"
            className={({ isActive }) =>
              `w-full flex items-center gap-4 rounded-xl p-4 mb-4 transition-all duration-300 ${
                isActive
                  ? "bg-blue-600"
                  : "hover:bg-slate-700"
              }`
            }
            >
            <BarChart3 size={22} />
            Analytics
          </NavLink>

          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `w-full flex items-center gap-4 rounded-xl p-4 transition-all duration-300 ${
                isActive
                  ? "bg-blue-600"
                  : "hover:bg-slate-700"
              }`
            }
            >
            <Settings size={22} />
            Settings
          </NavLink>

        </div>

      </div>

      {/* Footer */}

      <div className="p-6 border-t border-slate-700">

        <div className="bg-slate-700 rounded-xl p-4">

          <div className="flex items-center gap-3">

            <Activity
              className="text-green-400 animate-pulse"
              size={22}
            />

            <div>

              <h3 className="font-semibold">

                System Health

              </h3>

              <p className="text-green-400 text-sm">

                Running Normally

              </p>

            </div>

          </div>

        </div>

        <p className="text-gray-500 text-xs mt-5">

          Version 1.0.0

        </p>

      </div>

    </aside>
  );
}

export default Sidebar;