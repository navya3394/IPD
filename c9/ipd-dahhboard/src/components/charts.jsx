import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const threatData = [
  { time: "10:00", threats: 1 },
  { time: "10:05", threats: 2 },
  { time: "10:10", threats: 3 },
  { time: "10:15", threats: 2 },
  { time: "10:20", threats: 5 },
  { time: "10:25", threats: 3 },
];

const objectData = [
  { name: "Person", value: 75 },
  { name: "Knife", value: 15 },
  { name: "Weapon", value: 10 },
];

const COLORS = ["#3B82F6", "#F59E0B", "#EF4444"];

function Charts() {
  return (
    <div className="grid grid-cols-2 gap-6 mt-6">

      {/* Threat Trend */}

      <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6">

        <h2 className="text-xl font-bold mb-6">
          Threat Trend
        </h2>

        <ResponsiveContainer width="100%" height={280}>

          <LineChart data={threatData}>

            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />

            <XAxis dataKey="time" stroke="#94A3B8" />

            <YAxis stroke="#94A3B8" />

            <Tooltip />

            <Line
              type="monotone"
              dataKey="threats"
              stroke="#ef4444"
              strokeWidth={3}
            />

          </LineChart>

        </ResponsiveContainer>

      </div>

      {/* Object Distribution */}

      <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6">

        <h2 className="text-xl font-bold mb-6">
          Object Distribution
        </h2>

        <ResponsiveContainer width="100%" height={280}>

          <PieChart>

            <Pie
              data={objectData}
              dataKey="value"
              nameKey="name"
              outerRadius={90}
              label
            >

              {objectData.map((entry, index) => (
                <Cell
                  key={index}
                  fill={COLORS[index]}
                />
              ))}

            </Pie>

            <Tooltip />

          </PieChart>

        </ResponsiveContainer>

      </div>

    </div>
  );
}

export default Charts;