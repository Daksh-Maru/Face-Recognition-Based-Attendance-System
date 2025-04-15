import React, { useEffect, useState } from 'react';
// import axios from 'axios';

function Employee_Table() {
  const [attendanceData, setAttendanceData] = useState([]);

  useEffect(() => {
    // Uncomment and modify when your API is ready
    /*
    const fetchAttendance = async () => {
      try {
        const res = await axios.get('http://localhost:5000/api/attendance/all');
        setAttendanceData(res.data);
      } catch (error) {
        console.error('Failed to fetch attendance:', error);
      }
    };
    fetchAttendance();
    */

    // Mock Data
    const mockData = [
      { employeeNo: 'EMP001', timeIn: '2025-04-10T09:15:00' },
      { employeeNo: 'EMP002', timeIn: '2025-04-10T09:22:00' },
      { employeeNo: 'EMP001', timeIn: '2025-04-11T09:05:00' },
      { employeeNo: 'EMP002', timeIn: '2025-04-11T09:17:00' },
      { employeeNo: 'EMP001', timeIn: '2025-04-12T09:11:00' },
    ];

    setAttendanceData(mockData);
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Employee Attendance Table</h2>
      <table className="min-w-full border border-gray-300">
        <thead className="bg-gray-200">
          <tr>
            <th className="border px-4 py-2">Employee ID</th>
            <th className="border px-4 py-2">Time In</th>
          </tr>
        </thead>
        <tbody>
          {attendanceData.map((entry, index) => (
            <tr key={index} className="text-center">
              <td className="border px-4 py-2">{entry.employeeNo}</td>
              <td className="border px-4 py-2">
                {new Date(entry.timeIn).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Employee_Table;
