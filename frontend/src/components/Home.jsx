import React, { useState } from 'react'
import Header from './Header'
import Date_Picker from './Date_Picker'
import Attendance_Table from './Attendance_Table'

function Home() {
  const today = new Date().toISOString().split('T')[0];
  const [selectedDate, setSelectedDate] = useState(today);

  return (
    <div>
      <Header />
      <Date_Picker selectedDate={selectedDate} onDateChange={setSelectedDate} />
      <Attendance_Table selectedDate={selectedDate} />
    </div>
  )
}

export default Home