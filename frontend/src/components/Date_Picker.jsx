import React,{useState} from 'react'

function Date_Picker() {
    const today = new Date().toISOString().split('T')[0];
    const [selectedDate, setSelectedDate] = useState(today);

    const handleDateChange = (e) => {
        setSelectedDate(e.target.value);
    };
  return (
    <div className="flex flex-col items-center mt-6">
        <label className="mb-2 text-lg font-medium">Select a Date:</label>
        <input
            type="date"
            value={selectedDate}
            onChange={handleDateChange}
            className="border border-gray-300 p-2 rounded-md shadow-sm"
        />
        {selectedDate && (
            <p className="mt-4 text-gray-700">You selected: {selectedDate}</p>
        )}
    </div>
  )
}

export default Date_Picker