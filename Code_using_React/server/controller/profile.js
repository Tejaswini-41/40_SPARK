import User from '../models/User.js'; 

export const getUserProfile = async (req, res) => {
    try {
        const userId = req.user.id; 
        const user = await User.findById(userId).select('-password'); //exclude this pass
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }
        res.json(user); // Return user data
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Server error' });
    }
};
