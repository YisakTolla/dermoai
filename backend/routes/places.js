// backend/routes/places.js
const express = require('express');
const axios = require('axios');
const router = express.Router();

const GOOGLE_MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY;

// Geocoding endpoint
router.post('/geocode', async (req, res) => {
  try {
    const { address } = req.body;
    const response = await axios.get(
      `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(address)}&key=${GOOGLE_MAPS_API_KEY}`
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Geocoding failed' });
  }
});

// Places search endpoint
router.post('/places/nearby', async (req, res) => {
  try {
    const { location, radius, keyword, type } = req.body;
    const response = await axios.get(
      `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${location}&radius=${radius}&keyword=${keyword}&type=${type}&key=${GOOGLE_MAPS_API_KEY}`
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Places search failed' });
  }
});

// Place details endpoint
router.post('/places/details', async (req, res) => {
  try {
    const { placeId, fields } = req.body;
    const fieldsParam = fields ? `&fields=${fields.join(',')}` : '';
    const response = await axios.get(
      `https://maps.googleapis.com/maps/api/place/details/json?place_id=${placeId}${fieldsParam}&key=${GOOGLE_MAPS_API_KEY}`
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Place details failed' });
  }
});

module.exports = router;