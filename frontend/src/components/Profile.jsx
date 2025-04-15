import React, { useState,useEffect } from 'react'
import Employee_Table from './Employee_Table';
import Profile_content from './Profile_content';

function Profile() {
  return (
    <div>
        <Profile_content />
        <Employee_Table />
    </div>
  )
}

export default Profile