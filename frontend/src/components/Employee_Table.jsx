import React, { useEffect, useState } from 'react';
import api from '../api';

function Employee_Table({ employeeId }) {
  const [attendanceData, setAttendanceData] = useState([]);

  useEffect(() => {

    console.error(employeeId);
    if (!employeeId || isNaN(employeeId)) {
      console.error("Invalid employeeId:", employeeId);
      return; // skip fetch if invalid
    }

    const fetchAttendance = async () => {
      try {
        const res = await api.get(`attendance/employee/${ employeeId }`);
        console.log(res.data);
        setAttendanceData(res.data);
      } catch (error) {
        console.error('Failed to fetch attendance:', error);
      }
    };
    fetchAttendance();
  }, [employeeId]);
  

  return (
    <div className="p-4 text-white bg-">
      <h2 className="text-xl font-bold mb-4">Employee Attendance Records</h2>
      <table className="min-w-full border border-gray-300">
        <thead className="">
          <tr>
            <th className="border px-4 py-2">Sr. No</th>
            <th className="border px-4 py-2">Day & Date</th>
            <th className="border px-4 py-2">Time In</th>
          </tr>
        </thead>
          <tbody>
            {attendanceData.map((entry, index) => {
              const dateObj = new Date(entry.time_in);
              const day = dateObj.toLocaleDateString('en-US', { weekday: 'long' });
              const date = dateObj.toLocaleDateString();
              const time = dateObj.toLocaleTimeString();

              return (
                <tr key={index} className="hover:bg-gray-600">
                  <td className="border px-4 py-2">{index + 1}</td>
                  <td className="border px-4 py-2">{`${day}, ${date}`}</td>
                  <td className="border px-4 py-2">{time}</td>
                </tr>
              );
            })}
          </tbody>
      </table>
    </div>
  );
}

export default Employee_Table;
