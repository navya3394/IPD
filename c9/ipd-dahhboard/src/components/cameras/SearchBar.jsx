import {
  Search,
  Plus,
  Grid2X2,
  List,
  ChevronDown,
} from "lucide-react";

function SearchToolbar() {
  return (
    <div className="mb-8">

      {/* Header */}

      <div className="mb-6">

        <h1 className="text-4xl font-bold text-white">
          Cameras
        </h1>

        <p className="text-slate-400 mt-2 text-lg">
          Manage and monitor all surveillance cameras
        </p>

      </div>

      {/* Toolbar */}

      <div className="flex items-center justify-between">

        {/* Left Side */}

        <div className="flex items-center gap-4">

          {/* Search */}

          <div className="relative">

            <Search
              size={18}
              className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500"
            />

            <input
              type="text"
              placeholder="Search Camera..."
              className="
              w-[340px]
              h-12
              pl-11
              pr-4
              bg-slate-800
              border
              border-slate-700
              rounded-xl
              text-white
              placeholder:text-slate-500
              focus:outline-none
              focus:border-blue-500
              transition-all
              "
            />

          </div>

          {/* Status */}

          <button
            className="
            h-12
            px-5
            rounded-xl
            border
            border-slate-700
            bg-slate-800
            text-slate-300
            flex
            items-center
            gap-2
            hover:bg-slate-700
            transition-all
            "
          >

            Status

            <ChevronDown size={17} />

          </button>

        </div>

        {/* Right Side */}

        <div className="flex items-center gap-3">

          {/* Grid */}

          <button
            className="
            h-12
            w-12
            rounded-xl
            bg-blue-600
            flex
            items-center
            justify-center
            "
          >

            <Grid2X2 size={19} />

          </button>

          {/* List */}

          <button
            className="
            h-12
            w-12
            rounded-xl
            bg-slate-800
            border
            border-slate-700
            flex
            items-center
            justify-center
            hover:bg-slate-700
            transition-all
            "
          >

            <List size={19} />

          </button>

          {/* Add */}

          <button
            className="
            h-12
            px-6
            rounded-xl
            bg-blue-600
            hover:bg-blue-700
            transition-all
            flex
            items-center
            gap-2
            font-medium
            "
          >

            <Plus size={18} />

            Add Camera

          </button>

        </div>

      </div>

    </div>
  );
}

export default SearchToolbar;