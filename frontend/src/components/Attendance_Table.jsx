import React, { useEffect, useState } from 'react';
import axios from 'axios';

function Attendance_Table() {
  const today = new Date().toISOString().split('T')[0];
  const [selectedDate, setSelectedDate] = useState(today);
  const [attendanceData, setAttendanceData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get(`http://localhost:5000/api/attendance/${selectedDate}`);
        setAttendanceData(res.data);
      } catch (error) {
        console.error('Error fetching attendance data:', error);
        setAttendanceData([]);
      }
    };

    fetchData();
  }, [selectedDate]);

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mt-6 mb-4">Attendance for {selectedDate}</h2>
      <table className="min-w-full border border-gray-300">
        <thead className="bg-gray-200">
          <tr>
            <th className="border px-4 py-2">Sr. No</th>
            <th className="border px-4 py-2">Employee ID</th>
            <th className="border px-4 py-2">Name</th>
            <th className="border px-4 py-2">Time In</th>
          </tr>
        </thead>
        <tbody>
          {attendanceData.length > 0 ? (
            attendanceData.map((entry, index) => (
              <tr key={index} className="text-center">
                <td className="border px-4 py-2">{index + 1}</td>
                <td className="border px-4 py-2">{entry.employee_id}</td>
                <td className="border px-4 py-2">{entry.employee_name}</td>
                <td className="border px-4 py-2">
                  {new Date(entry.time_in).toLocaleString()}
                </td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan="4" className="text-center py-4">No attendance found for this date.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default Attendance_Table;
