import { Outlet } from "react-router-dom";

import Navbar from "../components/navbar";
import Sidebar from "../components/sidebar";

function MainLayout() {
  return (
    <div className="h-screen overflow-hidden bg-slate-900">

      <Navbar />

      <div className="flex h-[calc(100vh-80px)]">

        <Sidebar />

        <main className="flex-1 overflow-y-auto bg-slate-900">

          <Outlet />

        </main>

      </div>

    </div>
  );
}

export default MainLayout;