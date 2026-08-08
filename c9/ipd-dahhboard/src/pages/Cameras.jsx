import SearchToolbar from "../components/cameras/Searchbar";
import CameraCard from "../components/cameras/CameraCard";
import AddCameraCard from "../components/cameras/AddCameraCard";

function Cameras() {
  return (
    <div className="p-10">

      <div className="p-8">

        <SearchToolbar />

      </div>

      <div className="grid grid-cols-2 gap-6 mt-8">

        <CameraCard />

        <AddCameraCard />

        <AddCameraCard />

        <AddCameraCard />

      </div>

    </div>
  );
}

export default Cameras;