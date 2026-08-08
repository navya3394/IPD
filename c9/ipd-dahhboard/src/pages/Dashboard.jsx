import KPICards from "../components/KPI_cards";
import VideoFeed from "../components/Videofeed";
import ThreatStatus from "../components/Threat";
import LiveLogs from "../components/livelogs";
import Charts from "../components/charts";

function Dashboard() {
  return (
    <>

      <KPICards />

      <VideoFeed />

      <div className="grid grid-cols-12">

        <div className="col-span-4">

          <ThreatStatus />

        </div>

        <div className="col-span-8">

          <LiveLogs />

        </div>

      </div>

      <Charts />

    </>
  );
}

export default Dashboard;