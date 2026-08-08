import {
  Users,
  ShieldAlert,
  Camera,
  Cpu,
  TrendingUp,
} from "lucide-react";

function Card({
  title,
  value,
  subtitle,
  icon,
  valueColor = "text-white",
  iconBg,
  accent,
}) {
  return (
    <div
      className={`
        bg-slate-800
        border
        border-slate-700
        ${accent}
        h-32
        px-5
        py-4
        flex
        justify-between
        items-center
        transition-all
        duration-300
        hover:bg-slate-750
      `}
    >
      {/* Left Section */}

      <div>

        <p className="text-gray-400 text-base font-medium">
          {title}
        </p>

        <h1 className={`text-5xl font-bold leading-none mt-2 ${valueColor}`}>
          {value}
        </h1>

        <div className="mt-3 text-sm">
          {subtitle}
        </div>

      </div>

      {/* Right Section */}

      <div
        className={`
          ${iconBg}
          w-14
          h-14
          rounded-md
          flex
          items-center
          justify-center
        `}
      >
        {icon}
      </div>

    </div>
  );
}

function KPICards() {
  return (

    <div className="grid grid-cols-4 gap-[2px] bg-[#172033]">

      <Card
        title="People Detected"
        value="15"
        subtitle={
          <div className="flex items-center gap-1 text-green-400">
            <TrendingUp size={14} />
            <span>+12%</span>
          </div>
        }
        icon={<Users size={28} className="text-blue-400" />}
        iconBg="bg-blue-500/15"
        accent="border-t-[3px] border-t-blue-500"
      />

      <Card
        title="Threats"
        value="02"
        valueColor="text-red-500"
        subtitle={
          <span className="text-red-400">
            Active Alerts
          </span>
        }
        icon={<ShieldAlert size={28} className="text-red-400" />}
        iconBg="bg-red-500/15"
        accent="border-t-[3px] border-t-red-500"
      />

      <Card
        title="Cameras"
        value="01"
        subtitle={
          <span className="text-green-400">
            Online
          </span>
        }
        icon={<Camera size={28} className="text-green-400" />}
        iconBg="bg-green-500/15"
        accent="border-t-[3px] border-t-green-500"
      />

      <Card
        title="Processing FPS"
        value="30"
        subtitle={
          <span className="text-cyan-400">
            Stable
          </span>
        }
        icon={<Cpu size={28} className="text-cyan-400" />}
        iconBg="bg-cyan-500/15"
        accent="border-t-[3px] border-t-cyan-500"
      />

    </div>

  );
}

export default KPICards;