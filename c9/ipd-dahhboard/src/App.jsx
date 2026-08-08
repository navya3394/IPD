import { BrowserRouter, Routes, Route } from "react-router-dom";

import MainLayout from "./layout/MainLayout";

import Dashboard from "./pages/Dashboard";
import Cameras from "./pages/Cameras";
import ThreatAlerts from "./pages/ThreatAlerts";
import EventLogs from "./pages/EventLogs";
import Analytics from "./pages/Analytics";
import Settings from "./pages/Settings";

function App() {
  return (
    <BrowserRouter>

      <Routes>

        <Route path="/" element={<MainLayout />}>

          <Route index element={<Dashboard />} />

          <Route path="cameras" element={<Cameras />} />

          <Route path="threat-alerts" element={<ThreatAlerts />} />

          <Route path="event-logs" element={<EventLogs />} />

          <Route path="analytics" element={<Analytics />} />

          <Route path="settings" element={<Settings />} />

        </Route>

      </Routes>

    </BrowserRouter>
  );
}

export default App;