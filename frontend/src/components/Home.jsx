import React, {useState} from 'react'
import Header from './Header'
import Date_Picker from './Date_Picker'
import Attendance_Table from './Attendance_Table'

function Home() {
  return (
    <div>
        <Header />
        <Date_Picker />
        <Attendance_Table />
    </div>
  )
}

export default Home