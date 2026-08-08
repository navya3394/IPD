import { Plus } from "lucide-react";

function AddCameraCard() {
  return (
    <div className="bg-slate-800 border-2 border-dashed border-slate-700 rounded-2xl hover:border-blue-500 transition-all duration-300 cursor-pointer flex flex-col items-center justify-center h-full min-h-[560px]">

      <div className="w-20 h-20 rounded-full bg-blue-500/20 flex items-center justify-center">

        <Plus size={40} className="text-blue-400" />

      </div>

      <h2 className="text-white text-2xl font-semibold mt-6">

        Add Camera

      </h2>

      <p className="text-slate-400 mt-2">

        Connect a new surveillance camera

      </p>

    </div>
  );
}

export default AddCameraCard;