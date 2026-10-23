import { Link } from "../lib/route.jsx";

const DOORS = {
  explore: ["/explore", "For scientists", "Find the likely sites in your gene", "Explore sites →"],
  how: ["/how-it-works", "For the curious", "See how the model reaches a ranking", "How it works →"],
  stats: ["/model-stats", "For evaluating the model", "See how well the ranking holds up", "Model stats →"],
};

// The cards that close each page, pointing to the other pages.
export default function Doors({ to }) {
  return (
    <section className={`doors n${to.length}`}>
      {to.map((k) => {
        const [href, kicker, title, go] = DOORS[k];
        return (
          <Link key={k} to={href} className="door">
            <span className="door-kicker">{kicker}</span>
            <span className="door-title">{title}</span>
            <span className="door-go">{go}</span>
          </Link>
        );
      })}
    </section>
  );
}
