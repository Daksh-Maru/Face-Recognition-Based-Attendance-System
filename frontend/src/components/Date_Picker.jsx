import React, { useCallback } from 'react'

function Date_Picker({ selectedDate, onDateChange }) {
  const handleDateChange = useCallback((e) => {
    const newDate = e.target.value;
    if (newDate !== selectedDate) {
      onDateChange(newDate);
    }
  }, [selectedDate, onDateChange]);

  return (
    <div className="flex flex-col items-center mt-6">
      <label className="mb-2 text-lg font-medium">Select a Date:</label>
      <input
        type="date"
        value={selectedDate}
        onChange={handleDateChange}
        className="border border-gray-300 p-2 rounded-md shadow-sm"
        max={new Date().toISOString().split('T')[0]} // Prevent selecting future dates
      />
      {selectedDate && (
        <p className="mt-4 text-gray-400">You selected: {selectedDate}</p>
      )}
    </div>
  )
}

export default React.memo(Date_Picker);