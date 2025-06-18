import React, { useState, useEffect } from 'react';
import api from '../api';

function Profile_content({ employeeId }) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!employeeId) return;

    const fetchProfile = async () => {
      setLoading(true);
      try {
        // Fetch employee profile by ID
        const res = await api.get(`attendance/employee/${employeeId}`);
        // /attendance/employee/${ employeeId }

        // Assuming res.data is an array with at least one object
        if (res.data.length > 0) {
          const { employee_name, employee_id, image_path } = res.data[0];
          setProfile({
            name: employee_name,
            id: employee_id,
            image: image_path,
          });
        } else {
          setProfile(null);
        }
      } catch (error) {
        console.error('Error fetching employee profile:', error);
        setProfile(null);
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, [employeeId]);

  if (loading) {
    return <div className="p-4">Loading profile...</div>;
  }

  if (!profile) {
    return <div className="p-4">No profile found for employee ID: {employeeId}</div>;
  }

  return (
    <div className="p-4 flex flex-col items-center space-y-4">
      <img
        src={profile.image || ''}
        className="w-40 h-40 rounded-full border-2 border-gray-300 object-cover shadow-md"
        alt={`${profile.name}'s profile`}
      />
      <h2 className="text-2xl font-bold">{profile.name}</h2>
      <p className="text-gray-400 text-lg">Employee ID: {profile.id}</p>
    </div>
  );
}

export default Profile_content;
