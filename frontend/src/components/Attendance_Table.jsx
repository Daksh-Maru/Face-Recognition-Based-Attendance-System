import React, { useEffect, useState, useCallback } from 'react';
import api from '../api';
import { useNavigate } from 'react-router-dom';

function Attendance_Table({ selectedDate }) {
  const [attendanceData, setAttendanceData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const fetchData = useCallback(async (date) => {
    if (!date) return;
    
    setLoading(true);
    setError(null);
    try {
      const res = await api.get(`attendance/by-date/${date}`);
      // console.log(res.data);
      setAttendanceData(res.data);
    } catch (error) {
      console.error('Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status
      });
      setError(error.response?.data?.detail || 'Failed to fetch attendance or employee data');
      setAttendanceData([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      fetchData(selectedDate);
    }, 300); // Add a small delay to prevent rapid API calls

    return () => clearTimeout(timeoutId);
  }, [selectedDate, fetchData]);

  const formatTime = useCallback((timeString) => {
    if (!timeString) return '';
    try {
      const date = new Date(timeString);
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
    } catch (error) {
      console.error('Error formatting time:', error);
      return timeString;
    }
  }, []);

  if (!selectedDate) {
    return (
      <div className="p-6 text-center text-gray-700">
        Please select a date to view attendance records
      </div>
    );
  }

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mt-6 mb-4">Attendance for {selectedDate}</h2>
      {loading ? (
        <div className="text-center py-4">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          <p className="mt-2">Loading attendance data...</p>
        </div>
      ) : error ? (
        <div className="text-center py-4">
          <p className="text-red-500">{error}</p>
          <p className="text-sm text-gray-700 mt-2">Please try selecting a different date</p>
        </div>
      ) : (
        <table className="min-w-full border border-gray-300">
          <thead className="">
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
                <tr key={index} className="text-center hover:bg-gray-600" onClick={() => navigate(`/profile/${entry.employee_id}`)}>
                  <td className="border px-4 py-2">{index + 1}</td>
                  <td className="border px-4 py-2">
                  {entry.employee_id}
                  </td>
                  <td className="border px-4 py-2">{entry.employee_name}</td>
                  <td className="border px-4 py-2">
                    {formatTime(entry.time_in)}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="4" className="text-center py-4">No attendance records found for this date.</td>
              </tr>
            )}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default React.memo(Attendance_Table);
