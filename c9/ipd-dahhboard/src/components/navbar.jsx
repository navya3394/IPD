import { Wifi, Bell, ShieldCheck } from "lucide-react";
import { useState, useEffect } from "react";

function Navbar() {

  const [time,setTime]=useState(new Date());

  useEffect(()=>{

      const timer=setInterval(()=>{

          setTime(new Date());

      },1000);

      return ()=>clearInterval(timer);

  },[]);

  return(

    <nav className="h-20 bg-slate-900/90 backdrop-blur-md border-b border-slate-700 flex justify-between items-center px-8">

      <div>

        <h1 className="text-3xl font-bold">

          AI Surveillance Dashboard

        </h1>

        <p className="text-gray-400 text-sm">

          Smart Threat Detection System

        </p>

      </div>

      <div className="flex items-center gap-8">

        <div className="flex items-center gap-2 text-green-400">

          <Wifi size={18}/>

          Connected

        </div>

        <Bell className="cursor-pointer hover:text-blue-400"/>

        <ShieldCheck className="text-green-500"/>

        <div className="text-right">

          <h3>

            {time.toLocaleTimeString()}

          </h3>

          <p className="text-xs text-gray-400">

            {time.toDateString()}

          </p>

        </div>

      </div>

    </nav>

  )

}

export default Navbar;